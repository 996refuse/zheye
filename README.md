![image](https://raw.githubusercontent.com/996refuse/zheye/master/logo.png)

# 者也 - 知乎汉字验证码识别程序

知乎用户登录验证码很有个性，找到图中倒立文字的位置

https://www.zhihu.com/captcha.gif?type=login&lang=cn

![image](https://raw.githubusercontent.com/996refuse/zheye/master/zhihu.png)

# keyword

* 卷积神经网络
* 高斯混合模型

# online evaluate

在线测试地址 http://zheye.74ls74.org/

# how to use

```bash
# install dependency
pip install numpy pillow scikit-learn torch torchvision

# clone
git clone --depth=1 https://github.com/996refuse/zheye.git

# training (can skip)
python3 zheye/train.py

# testing
python3 zheye/evaluate.py realcap/01.gif
```

# license 

WTFPL
http://www.wtfpl.net/

本项目仅供学习娱乐，请勿滥用。请遵守知乎用户协议合理使用互联网

# donate

![image](https://raw.githubusercontent.com/996refuse/zheye/master/donate.png)
![image](https://raw.githubusercontent.com/996refuse/zheye/master/wechat.png)
