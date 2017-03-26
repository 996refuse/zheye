![image](https://raw.githubusercontent.com/muchrooms/zheye/master/logo.png)

# 者也 - 知乎汉字验证码识别程序

知乎新用户注册会要求识别一个古怪的验证码
https://www.zhihu.com/captcha.gif?r=1464247559114&type=login&lang=cn

此项目利用卷积神经网络识别验证码

在线测试地址 http://zheye.shidaixin.com

使用方法
```python
from zheye import util
util.Recognizing('./captcha.gif')
```

正确率80%左右,欢迎指教,欢迎任何批评建议

# WTFPL

```
            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                    Version 2, December 2004

 Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

 Everyone is permitted to copy and distribute verbatim or modified
 copies of this license document, and changing it is allowed as long
 as the name is changed.

            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. You just DO WHAT THE FUCK YOU WANT TO.
```
