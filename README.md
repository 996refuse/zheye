![image](https://raw.githubusercontent.com/muchrooms/zheye/master/logo.png)

# 者也 - 知乎汉字验证码识别程序

知乎新用户注册会要求识别一个古怪的验证码

https://www.zhihu.com/captcha.gif?type=login&lang=cn

![image](https://raw.githubusercontent.com/muchrooms/zheye/master/zhihu.png)

此项目利用高斯混合模型和卷积神经网络识别验证码

在线测试地址 http://zheye.shidaixin.com

使用方法

```python
from zheye import zheye
z = zheye()
positions = z.Recognize('path/to/captcha.gif')
```

验证集正确率92%左右,欢迎指教,欢迎任何批评建议

# WTFPL

http://www.wtfpl.net/
