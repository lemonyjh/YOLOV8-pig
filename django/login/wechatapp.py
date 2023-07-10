from django.http import HttpResponse

def pytoapp(request):
    signature = (request.GET.get('a'))
    timestamp = (request.GET.get('b'))
    nonce = (request.GET.get('b'))
    echostr = (request.GET.get('b'))
    m = "第一个变量为："+signature+",第二个变量为："+timestamp+",第三个变量为："+nonce+",第四个变量为："+echostr
    print(m)
    return HttpResponse(echostr)