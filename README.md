# ComfyUI_OOTDiffusion_CXH
OOTDiffusion

已解决：ComfyUI_UltimateSDUpscale 放大节点冲突 

支持hd + dc （训练数据集不一样）

hd:上身比较好

dc:主要是裤子+裙子效果好

安装教程:https://www.bilibili.com/video/BV1ew4m1f7bb/

模型位置：
![e2a8fbac2a5cd832f369ccd029f254a](https://github.com/StartHua/ComfyUI_OOTDiffusion_CXH/assets/22284244/96e3a676-1e62-4eba-bcd9-c8893f938600)


T-shir:
![470d3d1499d9b9ad571c3881f03a020](https://github.com/StartHua/ComfyUI_OOTDiffusion_CXH/assets/22284244/02680773-6086-42d3-ba31-d507dffbbde4)


dress:
![3f1f4ee6c85f59f04e320cbdafe2af6](https://github.com/StartHua/ComfyUI_OOTDiffusion_CXH/assets/22284244/719c4c19-5745-4431-b7f7-2fd847fbf8ad)


low:
![030ac69e80da49df979c457da125aa6](https://github.com/StartHua/ComfyUI_OOTDiffusion_CXH/assets/22284244/84f21e09-0d83-470a-a0cb-bb104b4594eb)


自己遇到的问题和解决方案紧参考(环境大家都不一样)：

（1）# RuntimeError: Ninja is required to load C++ extensions

pip install Ninja

如果存在warming提示就删除升级需要的lib再安装Ninja

需要把 ninja.exe在的目录添加到path

ComfyUI_windows_portable\python_embeded\Scripts

（2）.cl : Command line warning D9002 : ignoring unknown option '-O3'

 -O3改成 /O2

(3).Cannot open include file: 'Python.h':

用conda 新建立一个comfyui对应版本的python 拷贝Include 和libs到comfyui自带的python环境

忽略下面错误:

from modules_ import processing, shared, images, devices, scripts

ImportError: cannot import name 'processing' from 'modules' (xxxx\custom_nodes\ComfyUI_OOTDiffusion_CXH\preprocess\humanparsing\modules\__init__.py)


