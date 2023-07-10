import torch

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils.plotting import save_one_box
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.data.augment import classify_transforms
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode

import platform
from pathlib import Path
import cv2

import time

STREAM_WARNING = """
    WARNING ⚠️ stream/video/webcam/dir predict source will accumulate results in RAM unless `stream=True` is passed,
    causing potential out-of-memory errors for large sources or long-running streams/videos.

    Usage:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
"""


class BasePredictor:
    plotted_img: None

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        print(self.args)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)

        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.labeltxt = []
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

        self.labelcc = {'prone': ["趴卧", 0], 'stand': ["站立", 0], 'lateral': ["侧卧", 0],
                        'bin_empty': ["食槽无料", 0], 'bin_feed': ["食槽有料", 0], 'bin_unkown': ["食槽未知", 0]}

    def preprocess(self, img):
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0

        return img

    def write_results(self, idx, results, batch):
        p, im, im0 = batch

        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0

        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string

        result = results[idx]  # TODO: make boxes inherit from tensors
        if len(result) == 0:
            return f'{log_string}(no detections), '
        det = result.boxes
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        if self.args.save or self.args.show:  # Add bbox to image
            self.plotted_img = result.plot(line_width=self.args.line_thickness)

        # write
        for d in reversed(det):

            c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
            # print("数据：", int(d.cls))
            if self.args.save_txt:  # Write to file
                line = (c, *d.xywhn.view(-1)) + (conf,) * self.args.save_conf + (() if id is None else (id,))

                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        # cv2.imshow('image', self.plotted_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(log_string)

        return log_string

    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path, _, _, _, _ = self.batch
            # print(orig_img)

            img_path = path[i] if isinstance(path, list) else path
            # 书写标签的位置“Results”
            results.append(
                Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred, labels=self.labelcc))
        return results

    def __call__(self, source=None, model=None, stream=False):
        self.stream = stream
        if stream:
            return self.stream_inference(source, model)
        else:
            return list(self.stream_inference(source, model))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        # Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass
        return self.plotted_img, self.labeltxt, self.labelcc

    def setup_source(self, source):
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        if self.args.task == 'classify':
            transforms = getattr(self.model.model, 'transforms', classify_transforms(self.imgsz[0]))
        else:  # predict, segment
            transforms = None
        self.dataset = load_inference_source(source=source,
                                             transforms=transforms,
                                             imgsz=self.imgsz,
                                             vid_stride=self.args.vid_stride,
                                             stride=self.model.stride,
                                             auto=self.model.pt)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None):
        if self.args.verbose:
            LOGGER.info('')

        # setup model
        if not self.model:
            self.setup_model(model)
        # setup source every time predict is called

        self.setup_source(source if source is not None else self.args.source)

        # check if save_dir/ label file exists  创建文件夹
        # if self.args.save or self.args.save_txt:
        #     (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        # warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
        self.run_callbacks('on_predict_start')
        for batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im, im0s, vid_cap, s = batch

            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=False) if self.args.visualize else False

            # preprocess
            with self.dt[0]:
                im = self.preprocess(im)
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # inference
            with self.dt[1]:
                preds = self.model(im, augment=self.args.augment, visualize=visualize)

            # postprocess
            with self.dt[2]:
                self.results = self.postprocess(preds, im, im0s)

            self.run_callbacks('on_predict_postprocess_end')

            # visualize, save, write results
            n = len(im)
            for i in range(n):
                self.results[i].speed = {
                    'preprocess': self.dt[0].dt * 1E3 / n,
                    'inference': self.dt[1].dt * 1E3 / n,
                    'postprocess': self.dt[2].dt * 1E3 / n}
                if self.source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                    continue
                p, im0 = (path[i], im0s[i].copy()) if self.source_type.webcam or self.source_type.from_img \
                    else (path, im0s.copy())
                p = Path(p)
                self.txtshow(i)
                # print(self.labeltxt)

                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    s += self.write_results(i, self.results, (p, im, im0))

                if self.args.show and self.plotted_img is not None:
                    self.show(p)
                #
                # if self.args.save and self.plotted_img is not None:
                #     self.save_preds(vid_cap, i, str(self.save_dir / p.name))
            self.run_callbacks('on_predict_batch_end')
            yield from self.results

            # Print time (inference-only)
            if self.args.verbose:
                LOGGER.info(f'{s}{self.dt[1].dt * 1E3:.1f}ms')

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *self.imgsz)}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks('on_predict_end')

    def setup_model(self, model, verbose=True):
        device = select_device(self.args.device, verbose=verbose)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.model = AutoBackend(model,
                                 device=device,
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)
        self.device = device
        self.model.eval()

    def show(self, p):
        im0 = self.plotted_img
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[4].startswith('image') else 1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path):

        im0 = self.plotted_img
        # save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)

        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.vid_writer[idx].write(im0)

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """
        Add callback
        """
        self.callbacks[event].append(func)

    def imgshow(self):
        im0 = self.plotted_img
        return im0

    def txtshow(self, idx):

        result = self.results[idx]
        det = result.boxes
        for d in reversed(det):
            # print("数据：", int(d.cls))
            self.labeltxt.append(self.model.names[int(d.cls)])
            self.labelcc[self.model.names[int(d.cls)]][1] = self.labelcc[self.model.names[int(d.cls)]][1] + 1


def predict(source):
    t1 = time.time()
    model = 'pig.pt'
    args = dict(model=model, source=source)
    print('args',args)
    predictor = BasePredictor(overrides=args)
    print('predictor',predictor)
    img, b, c = predictor.predict_cli()
    l1 = ""
    for k, v in c.items():
        l1 = l1 + str(v[0]) + ":" + str(v[1]) + "\n"
    t2 = time.time()
    t = t2 - t1
    # print('L1:', l1)
    #
    # print("b:", b)
    # print("时间：%.2f" % t)
    return img,l1,t


if __name__ == '__main__':

    source = cv2.imread('pig3.jpg')
    img,text,time1 = predict(source)

    print("时间：%.2f" % time1)
    print(text)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()