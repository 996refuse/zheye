![image](https://raw.githubusercontent.com/muchrooms/zheye/master/logo.png)

# 者也 - 知乎汉字验证码识别程序

知乎用户登录会要求识别一个古怪的验证码

https://www.zhihu.com/captcha.gif?type=login&lang=cn

![image](https://raw.githubusercontent.com/muchrooms/zheye/master/zhihu.png)

此项目包含

* 字模生成训练集合 GenerateTrainingSet.ipynb
* 训练卷积神经网络 keras.ipynb
* 高斯混合模型切图 GMM.ipynb

在线测试地址 http://zheye.shidaixin.com

# 如何调用

安装依赖

```bash
git clone --depth=1 https://github.com/muchrooms/zheye.git
cd zheye
pip install -r requirements.txt
```

使用方法

```python
from zheye import zheye
z = zheye()
positions = z.Recognize('path/to/captcha.gif')
```

验证集精度99.4%, 召回率99.2%, 欢迎指教, 欢迎issue

# WTFPL

http://www.wtfpl.net/
