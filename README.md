# Awesome Super-Resolution

A curated list of awesome super-resolution resources.

Recently we released *[Deep Learning for Image Super-resolution: A Survey](https://arxiv.org/abs/1902.06068)* to the community. In this survey, we review this task on different aspects including problem statement, datasets, evaluation metrics, methodology, and domain-specific applications. Specifically, we decompose the state-of-the-art models into basic components (e.g., network design principles, learning strategies, etc), analyze these components hierarchically and further identify their advantages and limitations. We also raise some open issues and potential development directions in this field at the end of the survey.

After completing this survey, we decided to release the collected SR resources, hoping to push the development of the community. We will keep updating our survey and this SR resource collection. If you have any questions or suggestions, please feel free to contact us.

**Table of Contents**

* [1. Non-deep Learning based SR](#1-Non-deep-Learning-based-SR)
* [2. Supervised SR](#2-Supervised-SR)
    - [2.1 Generic Image SR](#21-Generic-Image-SR)
    - [2.2 Face Image SR](#22-Face-Image-SR)
    - [2.3 Video SR](#23-Video-SR)
    - [2.4 Domain-specific SR](#24-Domain-specific-SR)
* [3. Unsupervised SR](#3-Unsupervised-SR)
* [4. SR Datasets](#4-SR-Datasets)
* [5. SR Metrics](#5-SR-Metrics)
* [6. Survey Resources](#6-Survey-Resources)
* [7. Other Resources](#7-Other-Resources)

**Citing this work**

If this repository is helpful to you, please cite our [survey](https://arxiv.org/abs/1902.06068).

```
@article{wang2019deep,
    title={Deep learning for image super-resolution: A survey},
    author={Wang, Zhihao and Chen, Jian and Hoi, Steven CH},
    journal={arXiv preprint arXiv:1902.06068},
    year={2019}
}
```



## 1. Non-deep Learning based SR

**2013 ICCV**

1. **Anchored Neighborhood Regression for Fast Example-Based Super-Resolution**, *Timofte, Radu; De, Vincent; Van Gool, Luc*, [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2013/html/Timofte_Anchored_Neighborhood_Regression_2013_ICCV_paper.html)], [[Project](http://www.vision.ee.ethz.ch/~timofter/ICCV2013_ID1774_SUPPLEMENTARY/index.html)], `ANR`, `GR`
2. **Nonparametric Blind Super-resolution**, *Michaeli, Tomer; Irani, Michal*, [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2013/html/Michaeli_Nonparametric_Blind_Super-resolution_2013_ICCV_paper.html)], [[Project](http://www.wisdom.weizmann.ac.il/~vision/BlindSR.html)], `BlindSR`

**2014 ACCV**

1. **A+: Adjusted Anchored Neighborhood Regression for Fast Super-Resolution**, *Timofte, Radu; De Smet, Vincent; Van Gool, Luc*, [[ACCV](https://link.springer.com/chapter/10.1007/978-3-319-16817-3_8)], [[Project](http://www.vision.ee.ethz.ch/~timofter/ACCV2014_ID820_SUPPLEMENTARY/index.html)], `A+`

**2014 TIP**

1. **A Statistical Prediction Model Based on Sparse Representations for Single Image Super-Resolution**, *Peleg, Tomer; Elad, Michael*, [[Matlab*](https://elad.cs.technion.ac.il/software/?pn=1430)], [[TIP](https://ieeexplore.ieee.org/abstract/document/6739068)]

**2015 CVPR**

1. **Fast and accurate image upscaling with super-resolution forests**, *Schulter, Samuel; Leistner, Christian; Bischof, Horst*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2015/html/Schulter_Fast_and_Accurate_2015_CVPR_paper.html)], `RFL`
2. **Handling motion blur in multi-frame super-resolution**, *Ma, Ziyang; Liao, Renjie; Tao, Xin; Xu, Li; Jia, Jiaya; Wu, Enhua*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2015/html/Ma_Handling_Motion_Blur_2015_CVPR_paper.html)], [[Project](http://www.cse.cuhk.edu.hk/~leojia/projects/mfsr)]
3. **Single image super-resolution from transformed self-exemplars**, *Huang, Jia-Bin; Singh, Abhishek; Ahuja, Narendra*, [[Matlab*](https://github.com/jbhuang0604/SelfExSR)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2015/html/Huang_Single_Image_Super-Resolution_2015_CVPR_paper.html)], `SelfExSR`, `Urban100`

**2015 ICCV**

1. **Naive Bayes Super-Resolution Forest**, *Salvador, Jordi; Perez-Pellitero, Eduardo*, [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2015/html/Salvador_Naive_Bayes_Super-Resolution_ICCV_2015_paper.html)], [[Project](https://jordisalvador-image.blogspot.com/2015/08/iccv-2015.html)], `NBSRF`

**2016 CVPR**

1. **PSyCo: Manifold Span Reduction for Super Resolution**, *Perez-Pellitero, Eduardo; Salvador, Jordi; Ruiz-Hidalgo, Javier; Rosenhahn, Bodo*, [[C++/Matlab*](https://bitbucket.org/EduPerez/psycosuperres/wiki/Home)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2016/html/Perez-Pellitero_PSyCo_Manifold_Span_CVPR_2016_paper.html)], `PSyCo`

**2016 NIPS**

1. **Learning Parametric Sparse Models for Image Super-Resolution**, *Li, Yongbo; Dong, Weisheng; Xie, Xuemei; Shi, GUANGMING; Li, Xin; Xu, Donglai*, [[NIPS](http://papers.nips.cc/paper/6378-learning-parametric-sparse-models-for-image-super-resolution)]

**2017 CVIU**

1. **Learning a no-reference quality metric for single-image super-resolution**, *Ma, Chao; Yang, Chih-Yuan; Yang, Xiaokang; Yang, Ming-Hsuan*, [[arXiv](https://arxiv.org/abs/1612.05890)], [[CVIU](https://www.sciencedirect.com/science/article/pii/S107731421630203X)], [[Matlab*](https://github.com/chaoma99/sr-metric)], [[Project](https://sites.google.com/site/chaoma99/sr-metric)], `Ma`

**2017 CVPR**

1. **Simultaneous Super-Resolution and Cross-Modality Synthesis of 3D Medical Images Using Weakly-Supervised Joint Convolutional Sparse Coding**, *Huang, Yawen; Shao, Ling; Frangi, Alejandro F.*, [[arXiv](https://arxiv.org/abs/1705.02596)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Simultaneous_Super-Resolution_and_CVPR_2017_paper.html)], `WEENIE`

**2017 CVPRW**

1. **SRHRF+: Self-Example Enhanced Single Image Super-Resolution Using Hierarchical Random Forests**, *Huang, Jun-Jie; Liu, Tianrui; Dragotti, Pier Luigi; Stathaki, Tania*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Huang_SRHRF_Self-Example_Enhanced_CVPR_2017_paper.html)], `SRHRF+`

**2017 ICLR**

1. **Amortised MAP Inference for Image Super-resolution**, *Sønderby, Casper Kaae; Caballero, Jose; Theis, Lucas; Shi, Wenzhe; Huszár, Ferenc*, [[arXiv](https://arxiv.org/abs/1610.04490)], [[OpenReview](https://openreview.net/forum?id=S1RP6GLle)], `AffGAN`

**2017 TCI**

1. **RAISR: Rapid and Accurate Image Super Resolution**, *Romano, Yaniv; Isidoro, John; Milanfar, Peyman*, [[arXiv](https://arxiv.org/abs/1606.01299)], [[TCI](https://ieeexplore.ieee.org/document/7744595)], `RAISR`

**2018 CVPR**

1. **Fight Ill-Posedness with Ill-Posedness: Single-shot Variational Depth Super-Resolution from Shading**, *Haefner, Bjoern; Queau, Yvain; Mollenhoff, Thomas; Cremers, Daniel*, [[Matlab*](https://github.com/BjoernHaefner/DepthSRfromShading)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Haefner_Fight_Ill-Posedness_With_CVPR_2018_paper.html)]

**2018 IJCV**

1. **Hallucinating Compressed Face Images**, *Yang, Chih-Yuan; Liu, Sifei; Yang, Ming-Hsuan*, [[IJCV](https://link.springer.com/article/10.1007%2Fs11263-017-1044-4)], [[Matlab*](https://github.com/yangchihyuan/HallucinatingCompressedFaceImages)], [[Project](http://faculty.ucmerced.edu/mhyang/project/FHCI/)], `FHCI`



## 2. Supervised SR

### 2.1 Generic Image SR

**2014 ECCV**

1. **Learning a Deep Convolutional Network for Image Super-Resolution**, *Dong, Chao; Loy, Chen Change; He, Kaiming; Tang, Xiaoou*, [[ECCV](https://link-springer-com.libproxy.smu.edu.sg/chapter/10.1007/978-3-319-10593-2_13)], [[Project](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)], `SRCNN`

**2015 ICCV**

1. **Deep Networks for Image Super-Resolution with Sparse Prior**, *Wang, Zhaowen; Liu, Ding; Yang, Jianchao; Han, Wei; Huang, Thomas*, [[arXiv](https://arxiv.org/abs/1507.08905)], [[Matlab*](https://github.com/huangzehao/SCN_Matlab)], [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2015/html/Wang_Deep_Networks_for_ICCV_2015_paper.html)], [[Project](http://www.ifp.illinois.edu/~dingliu2/iccv15/)], `SCN`

**2016 CVPR**

1. **Accurate Image Super-Resolution Using Very Deep Convolutional Networks**, *Kim, Jiwon; Lee, Jung Kwon; Lee, Kyoung Mu*, [[arXiv](https://arxiv.org/abs/1511.04587)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2016/html/Kim_Accurate_Image_Super-resolution_CVPR_2016_paper.html)], [[Project](https://cv.snu.ac.kr/research/VDSR/)], `VDSR`
2. **Deeply-Recursive Convolutional Network for Image Super-Resolution**, *Kim, Jiwon; Lee, Jung Kwon; Lee, Kyoung Mu*, [[arXiv](https://arxiv.org/abs/1511.04491)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2016/html/Kim_Deeply-Recursive_Convolutional_Network_CVPR_2016_paper.html)], [[Project](https://cv.snu.ac.kr/research/DRCN/)], `DRCN`
3. **Seven ways to improve example-based single image super resolution**, *Timofte, Radu; Rothe, Rasmus; Van Gool, Luc*, [[arXiv](https://arxiv.org/abs/1511.02228)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2016/html/Timofte_Seven_Ways_to_CVPR_2016_paper.html)], [[Project](http://www.vision.ee.ethz.ch/~timofter/CVPR2016_ID769_SUPPLEMENTARY/index.html)], `IA`, `L20`

**2016 ECCV**

1. **Accelerating the Super-Resolution Convolutional Neural Network**, *Dong, Chao; Loy, Chen Change; Tang, Xiaoou*, [[arXiv](https://arxiv.org/abs/1608.00367)], [[ECCV](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_25)], [[Project](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)], `FSRCNN`, `General-100`
2. **Perceptual Losses for Real-Time Style Transfer and Super-Resolution**, *Johnson, Justin; Alahi, Alexandre; Fei-Fei, Li*, [[arXiv](https://arxiv.org/abs/1603.08155)], [[ECCV](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_43)], [[Project](https://cs.stanford.edu/people/jcjohns/eccv16/)], [[Torch*](https://github.com/jcjohnson/fast-neural-style)], `Perceptual loss`

**2016 ICLR**

1. **Super-Resolution with Deep Convolutional Sufficient Statistics**, *Bruna, Joan; Sprechmann, Pablo; LeCun, Yann*, [[arXiv](https://arxiv.org/abs/1511.05666)], [[ICLR](https://iclr.cc/archive/www/doku.php%3Fid=iclr2016:accepted-main.html)]

**2016 NIPS**

1. **Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections**, *Mao, Xiao-Jiao; Shen, Chunhua; Yang, Yu-Bin*, [[arXiv](https://arxiv.org/abs/1603.09056)], [[Caffe*](https://bitbucket.org/chhshen/image-denoising)], [[NIPS](http://papers.nips.cc/paper/6172-image-restoration-using-very-deep-convolutional-encoder-decoder-networks-with-symmetric-skip-connections)], `RED-Net`

**2016 TCI**

1. **Loss Functions for Neural Networks for Image Processing**, *Zhao, Hang; Gallo, Orazio; Frosio, Iuri; Kautz, Jan*, [[arXiv](https://arxiv.org/abs/1511.08861)], [[Caffe*](https://github.com/NVlabs/PL4NN)], [[Project](https://research.nvidia.com/publication/loss-functions-image-restoration-neural-networks)], [[TCI](https://ieeexplore.ieee.org/document/7797130)], `PL4NN`

**2016 TPAMI**

1. **Image Super-Resolution Using Deep Convolutional Networks**, *Dong, Chao; Loy, Chen Change; He, Kaiming; Tang, Xiaoou*, [[arXiv](https://arxiv.org/abs/1501.00092)], [[Project](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)], [[TPAMI](https://ieeexplore.ieee.org/document/7115171/)], `SRCNN`

**2016 WACV**

1. **Is Image Super-resolution Helpful for Other Vision Tasks?**, *Dai, Dengxin; Wang, Yujian; Chen, Yuhua; Van Gool, Luc*, [[arXiv](https://arxiv.org/abs/1509.07009)], [[WACV](https://ieeexplore.ieee.org/document/7477613)]

**2017 CVPR**

1. **Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution**, *Lai, Wei-Sheng; Huang, Jia-Bin; Ahuja, Narendra; Yang, Ming-Hsuan*, [[arXiv](https://arxiv.org/abs/1704.03915)], [[MatConvNet*](https://github.com/phoenix104104/LapSRN)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/html/Lai_Deep_Laplacian_Pyramid_CVPR_2017_paper.html)], [[Project](http://vllab.ucmerced.edu/wlai24/LapSRN)], `LapSRN`
2. **Image Super-Resolution via Deep Recursive Residual Network**, *Tai, Ying; Yang, Jian; Liu, Xiaoming*, [[Caffe*](https://github.com/tyshiwo/DRRN_CVPR17)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/html/Tai_Image_Super-Resolution_via_CVPR_2017_paper.html)], [[Project](http://cvlab.cse.msu.edu/project-super-resolution.html)], `DRRN`
3. **Learning Deep CNN Denoiser Prior for Image Restoration**, *Zhang, Kai; Zuo, Wangmeng; Gu, Shuhang; Zhang, Lei*, [[arXiv](https://arxiv.org/abs/1704.03264)], [[MatConvNet*](https://github.com/cszn/ircnn)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Learning_Deep_CNN_CVPR_2017_paper.html)], `IRCNN`
4. **Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network**, *Ledig, Christian; Theis, Lucas; Huszar, Ferenc; Caballero, Jose; Cunningham, Andrew; Acosta, Alejandro; Aitken, Andrew; Tejani, Alykhan; Totz, Johannes; Wang, Zehan; Shi, Wenzhe*, [[arXiv](https://arxiv.org/abs/1609.04802)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/html/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.html)], `SRGAN`, `SRResNet`

**2017 CVPRW**

1. **A Deep Convolutional Neural Network with Selection Units for Super-Resolution**, *Choi, Jae-Seok; Kim, Munchurl*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Choi_A_Deep_Convolutional_CVPR_2017_paper.html)], `SelNet`
2. **Balanced Two-Stage Residual Networks for Image Super-Resolution**, *Fan, Yuchen; Shi, Honghui; Yu, Jiahui; Liu, Ding; Han, Wei; Yu, Haichao; Wang, Zhangyang; Wang, Xinchao; Huang, Thomas S.*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Fan_Balanced_Two-Stage_Residual_CVPR_2017_paper.html)], `BTSRN`
3. **Beyond Deep Residual Learning for Image Restoration: Persistent Homology-Guided Manifold Simplification**, *Bae, Woong; Yoo, Jaejun; Ye, Jong Chul*, [[arXiv](https://arxiv.org/abs/1611.06345)], [[MatConvNet*](https://github.com/iorism/CNN)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Bae_Beyond_Deep_Residual_CVPR_2017_paper.html)]
4. **Deep Wavelet Prediction for Image Super-Resolution**, *Guo, Tiantong; Mousavi, Hojjat Seyed; Vu, Tiep Huu; Monga, Vishal*, [[Matlab*](https://github.com/tT0NG/DWSRx4)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Guo_Deep_Wavelet_Prediction_CVPR_2017_paper.html)], `DWSR`
5. **Enhanced Deep Residual Networks for Single Image Super-Resolution**, *Lim, Bee; Son, Sanghyun; Kim, Heewon; Nah, Seungjun; Lee, Kyoung Mu*, [[arXiv](https://arxiv.org/abs/1707.02921)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.html)], [[PyTorch*](https://github.com/thstkdgus35/EDSR-PyTorch)], [[Torch*](https://github.com/LimBee/NTIRE2017)], `EDSR`, `MDSR`
6. **Exploiting Reflectional and Rotational Invariance in Single Image Superresolution**, *Donn, Simon; Meeus, Laurens; Luong, Hiep Quang; Goossens, Bart; Philips, Wilfried*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Donne_Exploiting_Reflectional_and_CVPR_2017_paper.html)], `FSRCNN SEF + F`
7. **Fast and Accurate Image Super-Resolution Using a Combined Loss**, *Xu, Jinchang; Zhao, Yu; Dong, Yuan; Bai, Hongliang*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Xu_Fast_and_Accurate_CVPR_2017_paper.html)], `TLSR`
8. **FormResNet: Formatted Residual Learning for Image Restoration**, *Jiao, Jianbo; Tu, Wei-Chih; He, Shengfeng; Lau, Rynson W. H.*, [[MatConvNet*](https://bitbucket.org/JianboJiao/formresnet/)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Jiao_FormResNet_Formatted_Residual_CVPR_2017_paper.html)]
9. **Image Super Resolution Based on Fusing Multiple Convolution Neural Networks**, *Ren, Haoyu; El-Khamy, Mostafa; Lee, Jungwon*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Ren_Image_Super_Resolution_CVPR_2017_paper.html)], `CNF`
10. **NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study**, *Agustsson, Eirikur; Timofte, Radu*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Agustsson_NTIRE_2017_Challenge_CVPR_2017_paper.html)], [[Project](https://data.vision.ee.ethz.ch/cvl/DIV2K/)], `NTIRE`, `DIV2K`
11. **NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results**, *Timofte, Radu; Agustsson, Eirikur; Van Gool, Luc; Yang, Ming-Hsuan; Zhang, Lei; Lim, Bee; Son, Sanghyun; Kim, Heewon; Nah, Seungjun; Lee, Kyoung Mu; Wang, Xintao; Tian, Yapeng; Yu, Ke; Zhang, Yulun; Wu, Shixiang; Dong, Chao; Lin, Liang; Qiao, Yu; Loy, Chen Change; Bae, Woong; Yoo, Jaejun; Han, Yoseob; Ye, Jong Chul; Choi, Jae-Seok; Kim, Munchurl; Fan, Yuchen; Yu, Jiahui; Han, Wei; Liu, Ding; Yu, Haichao; Wang, Zhangyang; Shi, Honghui; Wang, Xinchao; Huang, Thomas S; Chen, Yunjin; Zhang, Kai; Zuo, Wangmeng; Tang, Zhimin; Luo, Linkai; Li, Shaohui; Fu, Min; Cao, Lei; Heng, Wen; Bui, Giang; Le, Truc; Duan, Ye; Tao, Dacheng; Wang, Ruxin; Lin, Xu; Pang, Jianxin; Xu, Jinchang; Zhao, Yu; Xu, Xiangyu; Pan, Jinshan; Sun, Deqing; Zhang, Yujin; Song, Xibin; Dai, Yuchao; Qin, Xueying; Huynh, Xuan-Phung; Guo, Tiantong; Mousavi, Hojjat Seyed; Vu, Tiep Huu; Monga, Vishal; Cruz, Cristovao; Egiazarian, Karen; Katkovnik, Vladimir; Mehta, Rakesh; Jain, Arnav Kumar; Agarwalla, Abhinav; Praveen, Ch V Sai; Zhou, Ruofan; Wen, Hongdiao; Zhu, Che; Xia, Zhiqiang; Wang, Zhengtao; Guo, Qi*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Timofte_NTIRE_2017_Challenge_CVPR_2017_paper.html)], [[Project](http://www.vision.ee.ethz.ch/ntire17/)], `NTIRE`

**2017 ICCV**

1. **EnhanceNet: Single Image Super-Resolution Through Automated Texture Synthesis**, *Sajjadi, Mehdi S. M.; Schölkopf, Bernhard; Hirsch, Michael*, [[arXiv](https://arxiv.org/abs/1612.07919)], [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2017/html/Sajjadi_EnhanceNet_Single_Image_ICCV_2017_paper.html)], [[TensorFlow*](https://github.com/msmsajjadi/EnhanceNet-Code)], `EnhanceNet`
2. **Image Super-Resolution Using Dense Skip Connections**, *Tong, Tong; Li, Gen; Liu, Xiejie; Gao, Qinquan*, [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2017/html/Tong_Image_Super-resolution_Using_ICCV_2017_paper.html)], `SRDenseNet`
3. **MemNet: A Persistent Memory Network for Image Restoration**, *Tai, Ying; Yang, Jian; Liu, Xiaoming; Xu, Chunyan*, [[arXiv](https://arxiv.org/abs/1708.02209)], [[Caffe*](https://github.com/tyshiwo/MemNet)], [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2017/html/Tai_MemNet_A_Persistent_ICCV_2017_paper.html)], `MemNet`
4. **Pixel Recursive Super Resolution**, *Dahl, Ryan; Norouzi, Mohammad; Shlens, Jonathon*, [[arXiv](https://arxiv.org/abs/1702.00783)], [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2017/html/Dahl_Pixel_Recursive_Super_ICCV_2017_paper.html)]

**2017 Pattern Recognition Letters**

1. **Convolutional Low-Resolution Fine-Grained Classification**, *Cai, Dingding; Chen, Ke; Qian, Yanlin; Kämäräinen, Joni-Kristian*, [[arXiv](https://arxiv.org/abs/1703.05393)], [[Caffe*](https://github.com/dingdingcai/RACNN)], [[Pattern Recognition Letters](https://www.sciencedirect.com/science/article/abs/pii/S0167865517303896)], `RACNN`

**2017 TIP**

1. **Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising**, *Zhang, Kai; Zuo, Wangmeng; Chen, Yunjin; Meng, Deyu; Zhang, Lei*, [[arXiv](https://arxiv.org/abs/1608.03981)], [[Keras/MatConvNet/PyTorch*](https://github.com/cszn/DnCNN)], [[TIP](https://ieeexplore.ieee.org/document/7839189/)], `DnCNN`

**2018 arXiv**

1. **Channel-wise and Spatial Feature Modulation Network for Single Image Super-Resolution**, *Hu, Yanting; Li, Jie; Huang, Yuanfei; Gao, Xinbo*, [[arXiv](https://arxiv.org/abs/1809.11130)], `CSFM`
2. **Deep Learning-based Image Super-Resolution Considering Quantitative and Perceptual Quality**, *Choi, Jun-Ho; Kim, Jun-Hyuk; Cheon, Manri; Lee, Jong-Seok*, [[arXiv](https://arxiv.org/abs/1809.04789)], [[TensorFlow*](https://github.com/idearibosome/tf-perceptual-eusr)], `4PP-EUSR`
3. **Dual Reconstruction Nets for Image Super-Resolution with Gradient Sensitive Loss**, *Guo, Yong; Chen, Qi; Chen, Jian; Huang, Junzhou; Xu, Yanwu; Cao, Jiezhang; Zhao, Peilin; Tan, Mingkui*, [[arXiv](https://arxiv.org/abs/1809.07099)], `DRN`
4. **Image Reconstruction with Predictive Filter Flow**, *Kong, Shu; Fowlkes, Charless*, [[arXiv](https://arxiv.org/abs/1811.11482)], [[Project](https://www.ics.uci.edu/~skong2/pff.html)], [[PyTorch*](https://github.com/aimerykong/predictive-filter-flow)]
5. **RAM: Residual Attention Module for Single Image Super-Resolution**, *Kim, Jun-Hyuk; Choi, Jun-Ho; Cheon, Manri; Lee, Jong-Seok*, [[arXiv](https://arxiv.org/abs/1811.12043)], `RAM, SRRAM`
6. **SREdgeNet: Edge Enhanced Single Image Super Resolution using Dense Edge Detection Network and Feature Merge Network**, *Kim, Kwanyoung; Chun, Se Young*, [[arXiv](https://arxiv.org/abs/1812.07174)], `SREdgeNet`
7. **Super-Resolution based on Image-Adapted CNN Denoisers: Incorporating Generalization of Training Data and Internal Learning in Test Time**, *Tirer, Tom; Giryes, Raja*, [[arXiv](https://arxiv.org/abs/1811.12866)], `IDBP`
8. **Task-Driven Super Resolution: Object Detection in Low-resolution Images**, *Haris, Muhammad; Shakhnarovich, Greg; Ukita, Norimichi*, [[arXiv](https://arxiv.org/abs/1803.11316)], `TDSR`
9. **Triple Attention Mixed Link Network for Single Image Super Resolution**, *Cheng, Xi; Li, Xiang; Yang, Jian*, [[arXiv](https://arxiv.org/abs/1810.03254)], `TAN`
10. **Unsupervised Degradation Learning for Single Image Super-Resolution**, *Zhao, Tianyu; Ren, Wenqi; Zhang, Changqing; Ren, Dongwei; Hu, Qinghua*, [[arXiv](https://arxiv.org/abs/1812.04240)], `DNSR`

**2018 CVPR**

1. **Deep Back-Projection Networks For Super-Resolution**, *Haris, Muhammad; Shakhnarovich, Greg; Ukita, Norimichi*, [[arXiv](https://arxiv.org/abs/1803.02735)], [[Caffe*](https://github.com/alterzero/DBPN-caffe)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Haris_Deep_Back-Projection_Networks_CVPR_2018_paper.html)], [[Project](https://www.toyota-ti.ac.jp/Lab/Denshi/iim/members/muhammad.haris/projects/DBPN.html)], [[PyTorch*](https://github.com/alterzero/DBPN-Pytorch)], `DBPN`
2. **Fast and Accurate Single Image Super-Resolution via Information Distillation Network**, *Hui, Zheng; Wang, Xiumei; Gao, Xinbo*, [[arXiv](https://arxiv.org/abs/1803.09454)], [[Caffe*](https://github.com/Zheng222/IDN-Caffe)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Hui_Fast_and_Accurate_CVPR_2018_paper.html)], [[TensorFlow*](https://github.com/Zheng222/IDN-tensorflow)], `IDN`
3. **Image Super-Resolution via Dual-State Recurrent Networks**, *Han, Wei; Chang, Shiyu; Liu, Ding; Yu, Mo; Witbrock, Michael; Huang, Thomas S.*, [[arXiv](https://arxiv.org/abs/1805.02704)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Han_Image_Super-Resolution_via_CVPR_2018_paper.html)], [[TensorFlow*](https://github.com/WeiHan3/dsrn)], `DSRN`
4. **Learning a Single Convolutional Super-Resolution Network for Multiple Degradations**, *Zhang, Kai; Zuo, Wangmeng; Zhang, Lei*, [[arXiv](https://arxiv.org/abs/1712.06116)], [[MatConvNet*](https://github.com/cszn/SRMD)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Learning_a_Single_CVPR_2018_paper.html)], [[Project](https://github.com/cszn/SRMD)], `SRMD`
5. **Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform**, *Wang, Xintao; Yu, Ke; Dong, Chao; Loy, Chen Change*, [[arXiv](https://arxiv.org/abs/1804.02815)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Recovering_Realistic_Texture_CVPR_2018_paper.html)], [[Project](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN)], [[PyTorch*](https://github.com/xinntao/BasicSR)], [[PyTorch*](https://github.com/xinntao/SFTGAN)], `SFT-GAN`, `OutdoorScene`
6. **Residual Dense Network for Image Super-Resolution**, *Zhang, Yulun; Tian, Yapeng; Kong, Yu; Zhong, Bineng; Fu, Yun*, [[arXiv](https://arxiv.org/abs/1802.08797)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Residual_Dense_Network_CVPR_2018_paper.html)], [[PyTorch*](https://github.com/yulunzhang/RDN)], `RDN`
7. **The Perception-Distortion Tradeoff**, *Blau, Yochai; Michaeli, Tomer*, [[arXiv](https://arxiv.org/abs/1711.06077)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)], [[Project](http://webee.technion.ac.il/people/tomermic/PerceptionDistortion/PD_tradeoff.htm)]

**2018 CVPRW**

1. **A Fully Progressive Approach to Single-Image Super-Resolution**, *Wang, Yifan; Perazzi, Federico; McWilliams, Brian; Sorkine-Hornung, Alexander; Sorkine-Hornung, Olga; Schroers, Christopher*, [[arXiv](https://arxiv.org/abs/1804.02900)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Wang_A_Fully_Progressive_CVPR_2018_paper.html)], [[PyTorch*](https://github.com/fperazzi/proSR)], `ProSR`
2. **Deep Residual Network with Enhanced Upscaling Module for Super-Resolution**, *Kim, Jun-Hyuk; Lee, Jong-Seok*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Kim_Deep_Residual_Network_CVPR_2018_paper.html)], [[TensorFlow*](https://github.com/junhyukk/EUSR-Tensorflow)], `EUSR`
3. **Efficient Module Based Single Image Super Resolution for Multiple Problems**, *Park, Dongwon; Kim, Kwanyoung; Chun, Se Young*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Park_Efficient_Module_Based_CVPR_2018_paper.html)], `EDSR-PP`, `EMBSR`
4. **Image Super-Resolution via Progressive Cascading Residual Network**, *Ahn, Namhyuk; Kang, Byungkon; Sohn, Kyung-Ah*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Ahn_Image_Super-Resolution_via_CVPR_2018_paper.html)], `Progressive CARN`
5. **IRGUN : Improved Residue Based Gradual Up-Scaling Network for Single Image Super Resolution**, *Sharma, Manoj; Mukhopadhyay, Rudrabha; Upadhyay, Avinash; Koundinya, Sriharsha; Shukla, Ankit; Chaudhury, Santanu*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Sharma_IRGUN_Improved_Residue_CVPR_2018_paper.html)], [[TensorFlow*](https://github.com/Rudrabha/8X-Super-Resolution)], `IRGUN`
6. **Large Receptive Field Networks for High-Scale Image Super-Resolution**, *Seif, George; Androutsos, Dimitrios*, [[arXiv](https://arxiv.org/abs/1804.08181)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Seif_Large_Receptive_Field_CVPR_2018_paper.html)], `LRFNet`
7. **Multi-level Wavelet-CNN for Image Restoration**, *Liu, Pengju; Zhang, Hongzhi; Zhang, Kai; Lin, Liang; Zuo, Wangmeng*, [[arXiv](https://arxiv.org/abs/1805.07071)], [[MatConvNet*](https://github.com/lpj0/MWCNN)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Liu_Multi-Level_Wavelet-CNN_for_CVPR_2018_paper.html)], `MWCNN`
8. **New Techniques for Preserving Global Structure and Denoising with Low Information Loss in Single-Image Super-Resolution**, *Bei, Yijie; Damian, Alex; Hu, Shijia; Menon, Sachit; Ravi, Nikhil; Rudin, Cynthia*, [[arXiv](https://arxiv.org/abs/1805.03383)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Bei_New_Techniques_for_CVPR_2018_paper.html)], [[PyTorch*](https://github.com/nikhilvravi/DukeSR)], [[TensorFlow*](https://github.com/websterbei/EDSR_tensorflow)], `ADRSR`, `DNSR`
9. **NTIRE 2018 Challenge on Single Image Super-Resolution: Methods and Results**, *Timofte, Radu; Gu, Shuhang; Wu, Jiqing; Van Gool, Luc*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Timofte_NTIRE_2018_Challenge_CVPR_2018_paper.html)], `NTIRE`
10. **Persistent Memory Residual Network for Single Image Super Resolution**, *Chen, Rong; Qu, Yanyun; Zeng, Kun; Guo, Jinkang; Li, Cuihua; Xie, Yuan*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Chen_Persistent_Memory_Residual_CVPR_2018_paper.html)], `MemEDSR`, `IRMem`

**2018 DICTA**

1. **Deep Bi-Dense Networks for Image Super-Resolution**, *Wang, Yucheng; Shen, Jialiang; Zhang, Jian*, [[arXiv](https://arxiv.org/abs/1810.04873)], [[DICTA](https://ieeexplore.ieee.org/document/8615817)], [[Torch*](https://github.com/JannaShen/DBDN)], `DBDN`

**2018 ECCV**

1. **CrossNet: An End-to-end Reference-based Super Resolution Network using Cross-scale Warping**, *Zheng, Haitian; Ji, Mengqi; Wang, Haoqian; Liu, Yebin; Fang, Lu*, [[arXiv](https://arxiv.org/abs/1807.10547)], [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Haitian_Zheng_CrossNet_An_End-to-end_ECCV_2018_paper.html)], [[PyTorch*](https://github.com/htzheng/ECCV2018_CrossNet_RefSR)], `CrossNet`
2. **Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network**, *Ahn, Namhyuk; Kang, Byungkon; Sohn, Kyung-Ah*, [[arXiv](https://arxiv.org/abs/1803.08664)], [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Namhyuk_Ahn_Fast_Accurate_and_ECCV_2018_paper.html)], [[PyTorch*](https://github.com/nmhkahn/CARN-pytorch)], `CARN`
3. **Image Super-Resolution Using Very Deep Residual Channel Attention Networks**, *Zhang, Yulun; Li, Kunpeng; Li, Kai; Wang, Lichen; Zhong, Bineng; Fu, Yun*, [[arXiv](https://arxiv.org/abs/1807.02758)], [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.html)], [[PyTorch*](https://github.com/yulunzhang/RCAN)], `RCAN`
4. **Multi-scale Residual Network for Image Super-Resolution**, *Li, Juncheng; Fang, Faming; Mei, Kangfu; Zhang, Guixu*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.html)], [[PyTorch*](https://github.com/MIVRC/MSRN-PyTorch)], `MSRN`
5. **SOD-MTGAN: Small Object Detection via Multi-Task Generative Adversarial Network**, *Bai, Yancheng; Zhang, Yongqiang; Ding, Mingli; Ghanem, Bernard*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Yongqiang_Zhang_SOD-MTGAN_Small_Object_ECCV_2018_paper.html)], `SOD-MTGAN`
6. **SRFeat: Single Image Super-Resolution with Feature Discrimination**, *Park, Seong-Jin; Son, Hyeongseok; Cho, Sunghyun; Hong, Ki-Sang; Lee, Seungyong*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper.html)], [[TensorFlow*](https://github.com/HyeongseokSon1/SRFeat)], `SRFeat`

**2018 ECCVW**

1. **Analyzing Perception-Distortion Tradeoff using Enhanced Perceptual Super-resolution Network**, *Vasu, Subeesh; Madam, Nimisha Thekke; N, Rajagopalan A.*, [[arXiv](https://arxiv.org/abs/1811.00344)], [[OpenAccess](http://openaccess.thecvf.com/content_eccv_2018_workshops/w25/html/Vasu_Analyzing_Perception-Distortion_Tradeoff_using_Enhanced_Perceptual_Super-resolution_Network_ECCVW_2018_paper.html)], [[PyTorch*](https://github.com/subeeshvasu/2018_subeesh_epsr_eccvw)], `EPSR`
2. **ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks**, *Wang, Xintao; Yu, Ke; Wu, Shixiang; Gu, Jinjin; Liu, Yihao; Dong, Chao; Loy, Chen Change; Qiao, Yu; Tang, Xiaoou*, [[arXiv](https://arxiv.org/abs/1809.00219)], [[OpenAccess](http://openaccess.thecvf.com/content_eccv_2018_workshops/w25/html/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.html)], [[PyTorch*](https://github.com/xinntao/BasicSR)], [[PyTorch*](https://github.com/xinntao/ESRGAN)], `ESRGAN`
3. **Generative adversarial network-based image super-resolution using perceptual content losses**, *Cheon, Manri; Kim, Jun-Hyuk; Choi, Jun-Ho; Lee, Jong-Seok*, [[arXiv](https://arxiv.org/abs/1809.04783)], [[OpenAccess](http://openaccess.thecvf.com/content_eccv_2018_workshops/w25/html/Cheon_Generative_Adversarial_Network-based_Image_Super-Resolution_using_Perceptual_Content_Losses_ECCVW_2018_paper.html)], [[TensorFlow*](https://github.com/manricheon/eusr-pcl-tf)], `EUSR-PCL`
4. **The 2018 PIRM Challenge on Perceptual Image Super-resolution**, *Blau, Yochai; Mechrez, Roey; Timofte, Radu; Michaeli, Tomer; Zelnik-Manor, Lihi*, [[arXiv](https://arxiv.org/abs/1809.07517)], [[Matlab*](https://github.com/roimehrez/PIRM2018)], [[OpenAccess](http://openaccess.thecvf.com/content_eccv_2018_workshops/w25/html/Blau_2018_PIRM_Challenge_on_Perceptual_Image_Super-resolution_ECCVW_2018_paper.html)], [[Project1](https://pirm.github.io/)], [[Project2](https://www.pirm2018.org/PIRM-SR.html)], `PIRM`

**2018 NIPS**

1. **Joint Sub-bands Learning with Clique Structures for Wavelet Domain Super-Resolution**, *Zhong, Zhisheng; Shen, Tiancheng; Yang, Yibo; Lin, Zhouchen; Zhang, Chao*, [[arXiv](https://arxiv.org/abs/1809.04508)], [[NIPS](http://papers.nips.cc/paper/7301-joint-sub-bands-learning-with-clique-structures-for-wavelet-domain-super-resolution)], `SRCliqueNet`
2. **Non-Local Recurrent Network for Image Restoration**, *Liu, Ding; Wen, Bihan; Fan, Yuchen; Loy, Chen Change; Huang, Thomas S*, [[arXiv](https://arxiv.org/abs/1806.02919)], [[NIPS](http://papers.nips.cc/paper/7439-non-local-recurrent-network-for-image-restoration)], [[TensorFlow*](https://github.com/Ding-Liu/NLRN)], `NLRN`

**2018 TPAMI**

1. **Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks**, *Lai, Wei-Sheng; Huang, Jia-Bin; Ahuja, Narendra; Yang, Ming-Hsuan*, [[arXiv](https://arxiv.org/abs/1710.01992)], [[MatConvNet*](https://github.com/phoenix104104/LapSRN)], [[Project](http://vllab.ucmerced.edu/wlai24/LapSRN/)], [[TPAMI](https://ieeexplore.ieee.org/document/8434354)], `MS-LapSRN`

**2019 arXiv**

1. **A Matrix-in-matrix Neural Network for Image Super Resolution**, *Ma, Hailong; Chu, Xiangxiang; Zhang, Bo; Wan, Shaohua; Zhang, Bo*, [[arXiv](https://arxiv.org/abs/1903.07949)], [[PyTorch*](https://github.com/macn3388/MCAN)], `MCAN`
2. **Deep Back-Projection Networks for Single Image Super-resolution**, *Haris, Muhammad; Shakhnarovich, Greg; Ukita, Norimichi*, [[arXiv](https://arxiv.org/abs/1904.05677)], [[Caffe*](https://github.com/alterzero/DBPN-caffe)], [[PyTorch*](https://github.com/alterzero/DBPN-Pytorch)], [[Pytorch](https://github.com/thstkdgus35/EDSR-PyTorch)], `DBPN`
3. **Fast, Accurate and Lightweight Super-Resolution with Neural Architecture Search**, *Chu, Xiangxiang; Zhang, Bo; Ma, Hailong; Xu, Ruijun; Li, Jixiang; Li, Qingyuan*, [[arXiv](https://arxiv.org/abs/1901.07261)], [[TensorFlow*](https://github.com/falsr/FALSR)], `FALSR`
4. **Photo-realistic Image Super-resolution with Fast and Lightweight Cascading Residual Network**, *Ahn, Namhyuk; Kang, Byungkon; Sohn, Kyung-Ah*, [[arXiv](https://arxiv.org/abs/1903.02240)], [[PyTorch*](https://github.com/nmhkahn/PCARN-pytorch)], `PCARN`

**2019 CVPR**

1. **Blind Super-Resolution With Iterative Kernel Correction**, *Gu, Jinjin; Lu, Hannan; Zuo, Wangmeng; Dong, Chao*, [[arXiv](https://arxiv.org/abs/1904.03377)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Gu_Blind_Super-Resolution_With_Iterative_Kernel_Correction_CVPR_2019_paper.html)], `IKC`
2. **Camera Lens Super-Resolution**, *Chen, Chang; Xiong, Zhiwei; Tian, Xinmei; Zha, Zheng-Jun; Wu, Feng*, [[arXiv](https://arxiv.org/abs/1904.03378)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Camera_Lens_Super-Resolution_CVPR_2019_paper.html)], [[TensorFlow*](https://github.com/ngchc/CameraSR)], `CameraSR`, `City100`
3. **Deep Network Interpolation for Continuous Imagery Effect Transition**, *Wang, Xintao; Yu, Ke; Dong, Chao; Tang, Xiaoou; Loy, Chen Change*, [[arXiv](https://arxiv.org/abs/1811.10515)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Deep_Network_Interpolation_for_Continuous_Imagery_Effect_Transition_CVPR_2019_paper.html)], [[Project](https://xinntao.github.io/projects/DNI)], [[PyTorch*](https://github.com/xinntao/DNI)], `DNI`
4. **Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels**, *Zhang, Kai; Zuo, Wangmeng; Zhang, Lei*, [[arXiv](https://arxiv.org/abs/1903.12529)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Deep_Plug-And-Play_Super-Resolution_for_Arbitrary_Blur_Kernels_CVPR_2019_paper.html)], [[PyTorch*](https://github.com/cszn/DPSR)], `DPSR`
5. **Feedback Network for Image Super-Resolution**, *Li, Zhen; Yang, Jinglei; Liu, Zheng; Yang, Xiaomin; Jeon, Gwanggil; Wu, Wei*, [[arXiv](https://arxiv.org/abs/1903.09814)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Feedback_Network_for_Image_Super-Resolution_CVPR_2019_paper.html)], [[PyTorch*](https://github.com/Paper99/SRFBN_CVPR19)], `SRFBN`
6. **Image Super-Resolution by Neural Texture Transfer**, *Zhang, Zhifei; Wang, Zhaowen; Lin, Zhe; Qi, Hairong*, [[arXiv](https://arxiv.org/abs/1903.00834)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Image_Super-Resolution_by_Neural_Texture_Transfer_CVPR_2019_paper.html)], [[Project](http://web.eecs.utk.edu/~zzhang61/project_page/SRNTT/SRNTT.html)], [[TensorFlow*](https://github.com/ZZUTK/SRNTT)], `SRNTT`, `CUFED5`
7. **Meta-SR: A Magnification-Arbitrary Network for Super-Resolution**, *Hu, Xuecai; Mu, Haoyuan; Zhang, Xiangyu; Wang, Zilei; Sun, Jian; Tan, Tieniu*, [[arXiv](https://arxiv.org/abs/1903.00875)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Hu_Meta-SR_A_Magnification-Arbitrary_Network_for_Super-Resolution_CVPR_2019_paper.html)], [[PyTorch*](https://github.com/XuecaiHu/Meta-SR-Pytorch)], `Meta-SR`
8. **Natural and Realistic Single Image Super-Resolution With Explicit Natural Manifold Discrimination**, *Woong Soh, Jae; Yong Park, Gu; Jo, Junho; Ik Cho, Nam*, [[Code*](https://github.com/JWSoh/NatSR)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Soh_Natural_and_Realistic_Single_Image_Super-Resolution_With_Explicit_Natural_Manifold_CVPR_2019_paper.html)], `NatSR`
9. **ODE-Inspired Network Design for Single Image Super-Resolution**, *Xiangyu, He; Zitao, Mo; Peisong, Wang; Yang, Liu; Mingyuan, Yang; Jian, Cheng*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/He_ODE-Inspired_Network_Design_for_Single_Image_Super-Resolution_CVPR_2019_paper.html)], [[PyTorch*](https://github.com/HolmesShuan/OISR-PyTorch)], `ODE`
10. **Residual Networks for Light Field Image Super-Resolution**, *Zhang, Shuo; Lin, Youfang; Sheng, Hao*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Residual_Networks_for_Light_Field_Image_Super-Resolution_CVPR_2019_paper.html)]
11. **Second-order Attention Network for Single Image Super-Resolution**, *Dai, Tao; Cai, Jianrui; Zhang, Yongbing; Xia, Shu-Tao; Zhang, Lei*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.html)], [[PyTorch*](https://github.com/daitao/SAN)], `SAN`
12. **Towards Real Scene Super-Resolution with Raw Images**, *Xu, Xiangyu; Ma, Yongrui; Sun, Wenxiu*, [[arXiv](https://arxiv.org/abs/1905.12156)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Towards_Real_Scene_Super-Resolution_With_Raw_Images_CVPR_2019_paper.html)], [[Project](https://sites.google.com/view/xiangyuxu/rawsr_cvpr19)]
13. **Zoom to Learn, Learn to Zoom**, *Zhang, Xuaner; Chen, Qifeng; Ng, Ren; Koltun, Vladlen*, [[arXiv](https://arxiv.org/abs/1905.05169)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Zoom_to_Learn_Learn_to_Zoom_CVPR_2019_paper.html)], [[Project1](https://ceciliavision.github.io/project-pages/project-zoom.html)], [[Project2](http://vladlen.info/publications/zoom-learn-learn-zoom/)], [[TensorFlow*](https://github.com/ceciliavision/zoom-learn-zoom)], [[Video](https://www.youtube.com/watch?v=if6hZKglgL0)], `SR-RAW`, `CoBi`

**2019 CVPRW**

1. **An Epipolar Volume Autoencoder With Adversarial Loss for Deep Light Field Super-Resolution**, *Zhu, Minchen; Alperovich, Anna; Johannsen, Ole; Sulc, Antonin; Goldluecke, Bastian*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Zhu_An_Epipolar_Volume_Autoencoder_With_Adversarial_Loss_for_Deep_Light_CVPRW_2019_paper.html)], `DiffWGAN`
2. **DenseNet With Deep Residual Channel-Attention Blocks for Single Image Super Resolution**, *Jang, Dong-Won; Park, Rae-Hong*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Jang_DenseNet_With_Deep_Residual_Channel-Attention_Blocks_for_Single_Image_Super_CVPRW_2019_paper.html)], [[PyTorch*](https://github.com/dong-won-jang/DRCA)], `DRCA`
3. **Encoder-Decoder Residual Network for Real Super-Resolution**, *Cheng, Guoan; Matsune, Ai; Li, Qiuyu; Zhu, Leilei; Zang, Huaijuan; Zhan, Shu*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Cheng_Encoder-Decoder_Residual_Network_for_Real_Super-Resolution_CVPRW_2019_paper.html)], [[PyTorch*](https://github.com/yyknight/NTIRE2019_EDRN)], `EDRN`
4. **Fractal Residual Network and Solutions for Real Super-Resolution**, *Kwak, Junhyung; Son, Donghee*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Kwak_Fractal_Residual_Network_and_Solutions_for_Real_Super-Resolution_CVPRW_2019_paper.html)], `FRN`
5. **Hierarchical Back Projection Network for Image Super-Resolution**, *Liu, Zhi-Song; Wang, Li-Wen; Li, Chu-Tak; Siu, Wan-Chi*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Liu_Hierarchical_Back_Projection_Network_for_Image_Super-Resolution_CVPRW_2019_paper.html)], `HBPN`
6. **Light Field Super-Resolution: A Benchmark**, *Cheng, Zhen; Xiong, Zhiwei; Chen, Chang; Liu, Dong*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Cheng_Light_Field_Super-Resolution_A_Benchmark_CVPRW_2019_paper.html)]
7. **Multi-Scale Deep Neural Networks for Real Image Super-Resolution**, *Gao, Shangqi; Zhuang, Xiahai*, [[arXiv](https://arxiv.org/abs/1904.10698)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Gao_Multi-Scale_Deep_Neural_Networks_for_Real_Image_Super-Resolution_CVPRW_2019_paper.html)], [[TensorFlow*](https://github.com/shangqigao/gsq-image-SR)], `MsDNN`
8. **NTIRE 2019 Challenge on Real Image Super-Resolution: Methods and Results**, *Cai, Jianrui; Gu, Shuhang; Timofte, Radu; Zhang, Lei*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Cai_NTIRE_2019_Challenge_on_Real_Image_Super-Resolution_Methods_and_Results_CVPRW_2019_paper.html)], `NTIRE`
9. **Orientation-Aware Deep Neural Network for Real Image Super-Resolution**, *Du, Chen; Zewei, He; Anshun, Sun; Jiangxin, Yang; Yanlong, Cao; Yanpeng, Cao; Siliang, Tang; Yang, Michael Ying*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Du_Orientation-Aware_Deep_Neural_Network_for_Real_Image_Super-Resolution_CVPRW_2019_paper.html)], `OA-DNN`
10. **SCAN: Spatial Color Attention Networks for Real Single Image Super-Resolution**, *Xu, Xuan; Li, Xin*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Xu_SCAN_Spatial_Color_Attention_Networks_for_Real_Single_Image_Super-Resolution_CVPRW_2019_paper.html)], `SCAN`
11. **Suppressing Model Overfitting for Image Super-Resolution Networks**, *Feng, Ruicheng; Gu, Jinjin; Qiao, Yu; Dong, Chao*, [[arXiv](https://arxiv.org/abs/1906.04809)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Feng_Suppressing_Model_Overfitting_for_Image_Super-Resolution_Networks_CVPRW_2019_paper.html)]

**2019 ICLR**

1. **RESIDUAL NON-LOCAL ATTENTION NETWORKS FOR IMAGE RESTORATION**, *Zhang, Yulun; Li, Kunpeng; Li, Kai; Zhong, Bineng; Fu, Yun*, [[arXiv](https://arxiv.org/abs/1903.10082)], [[OpenReview](https://openreview.net/forum?id=HkeGhoA5FX)], [[PyTorch*](https://github.com/yulunzhang/RNAN)], `RNAN`

**2019 TPAMI**

1. **Toward Bridging the Simulated-to-Real Gap: Benchmarking Super-Resolution on Real Data**, *Kohler, Thomas; Batz, Michel; Naderi, Farzad; Kaup, Andre; Maier, Andreas; Riess, Christian*, [[Matlab*](https://github.com/thomas-koehler/SupER)], [[Project](https://superresolution.tf.fau.de/)], [[TPAMI](https://ieeexplore.ieee.org/document/8716546)], `SupER`

### 2.2 Face Image SR

**2016 ECCV**

1. **Deep Cascaded Bi-Network for Face Hallucination**, *Zhu, Shizhan; Liu, Sifei; Loy, Chen Change; Tang, Xiaoou*, [[arXiv](https://arxiv.org/abs/1607.05046)], [[Caffe*](https://github.com/zhusz/ECCV16-CBN)], [[ECCV](https://link.springer.com/chapter/10.1007/978-3-319-46454-1_37)], `CBN`
2. **Ultra-Resolving Face Images by Discriminative Generative Networks**, *Yu, Xin; Porikli, Fatih*, [[ECCV](https://link.springer.com/chapter/10.1007/978-3-319-46454-1_20)], [[Torch*](https://github.com/XinYuANU/URDGN)], `UR-DGN`

**2017 AAAI**

1. **Face Hallucination with Tiny Unaligned Images by Transformative Discriminative Neural Networks**, *Yu, Xin; Porikli, Fatih*, [[AAAI](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14340)], [[Torch*](https://github.com/XinYuANU/TDN)], `TDN`

**2017 CVPR**

1. **Attention-Aware Face Hallucination via Deep Reinforcement Learning**, *Cao, Qingxing; Lin, Liang; Shi, Yukai; Liang, Xiaodan; Li, Guanbin*, [[arXiv](https://arxiv.org/abs/1708.03132)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/html/Cao_Attention-Aware_Face_Hallucination_CVPR_2017_paper.html)], [[Torch*](https://github.com/ykshi/facehallucination)], `Attention-FH`
2. **Hallucinating Very Low-Resolution Unaligned and Noisy Face Images by Transformative Discriminative Autoencoders**, *Yu, Xin; Porikli, Fatih*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/html/Yu_Hallucinating_Very_Low-Resolution_CVPR_2017_paper.html)], [[Torch*](https://github.com/XinYuANU/TDAE)], `TDAE`

**2017 ICCV**

1. **Learning to Super-Resolve Blurry Face and Text Images**, *Xu, Xiangyu; Sun, Deqing; Pan, Jinshan; Zhang, Yujin; Pfister, Hanspeter; Yang, Ming-Hsuan*, [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2017/html/Xu_Learning_to_Super-Resolve_ICCV_2017_paper.html)], [[Project](https://sites.google.com/view/xiangyuxu/deblursr_iccv17)], `MCGAN`, `SCGAN`
2. **Wavelet-SRNet: A Wavelet-Based CNN for Multi-scale Face Super Resolution**, *Huang, Huaibo; He, Ran; Sun, Zhenan; Tan, Tieniu*, [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2017/html/Huang_Wavelet-SRNet_A_Wavelet-Based_ICCV_2017_paper.html)], [[PyTorch*](https://github.com/hhb072/WaveletSRNet)], `Wavelet-SRNet`

**2017 IJCAI**

1. **Learning to Hallucinate Face Images via Component Generation and Enhancement**, *Song, Yibing; Zhang, Jiawei; He, Shengfeng; Bao, Linchao; Yang, Qingxiong*, [[arXiv](https://arxiv.org/abs/1708.00223)], [[IJCAI](https://dl.acm.org/citation.cfm?id=3171921)], [[Project](https://ybsong00.github.io/ijcai17_sr/index.html)], `LCGE`

**2018 CVPR**

1. **FSRNet: End-to-End Learning Face Super-Resolution with Facial Priors**, *Chen, Yu; Tai, Ying; Liu, Xiaoming; Shen, Chunhua; Yang, Jian*, [[arXiv](https://arxiv.org/abs/1711.10703)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Chen_FSRNet_End-to-End_Learning_CVPR_2018_paper.html)], [[Torch*](https://github.com/tyshiwo/FSRNet)], `FSRNet`
2. **Super-FAN: Integrated facial landmark localization and super-resolution of real-world low resolution faces in arbitrary poses with GANs**, *Bulat, Adrian; Tzimiropoulos, Georgios*, [[arXiv](https://arxiv.org/abs/1712.02765)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Bulat_Super-FAN_Integrated_Facial_CVPR_2018_paper.html)], `Super-FAN`
3. **Super-Resolving Very Low-Resolution Face Images with Supplementary Attributes**, *Yu, Xin; Fernando, Basura; Hartley, Richard; Porikli, Fatih*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Super-Resolving_Very_Low-Resolution_CVPR_2018_paper.html)], [[Torch*](https://github.com/XinYuANU/FaceAttr)], `FaceAttr`

**2018 CVPRW**

1. **Attribute Augmented Convolutional Neural Network for Face Hallucination**, *Lee, Cheng-Han; Zhang, Kaipeng; Lee, Hu-Cheng; Cheng, Chia-Wen; Hsu, Winston*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Lee_Attribute_Augmented_Convolutional_CVPR_2018_paper.html)], [[TensorFlow*](https://github.com/steven413d/AACNN)], `AACNN`

**2018 ECCV**

1. **Face Super-Resolution Guided by Facial Component Heatmaps**, *Yu, Xin; Fernando, Basura; Ghanem, Bernard; Porikli, Fatih; Hartley, Richard*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Xin_Yu_Face_Super-resolution_Guided_ECCV_2018_paper.html)], `MTUN`
2. **Super-Identity Convolutional Neural Network for Face Hallucination**, *Zhang, Kaipeng; Zhang, Zhanpeng; Cheng, Chia-Wen; Hsu, Winston H.; Qiao, Yu; Liu, Wei; Zhang, Tong*, [[arXiv](https://arxiv.org/abs/1811.02328)], [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Kaipeng_Zhang_Super-Identity_Convolutional_Neural_ECCV_2018_paper.html)], `SICNN`

**2019 CVPRW**

1. **Exemplar Guided Face Image Super-Resolution Without Facial Landmarks**, *Dogan, Berk; Gu, Shuhang; Timofte, Radu*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Dogan_Exemplar_Guided_Face_Image_Super-Resolution_Without_Facial_Landmarks_CVPRW_2019_paper.html)], `GWAInet`

### 2.3 Video SR

**2015 ICCV**

1. **Video Super-Resolution via Deep Draft-Ensemble Learning**, *Liao, Renjie; Tao, Xin; Li, Ruiyu; Ma, Ziyang; Jia, Jiaya*, [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2015/html/Liao_Video_Super-Resolution_via_ICCV_2015_paper.html)], [[Project](http://www.cse.cuhk.edu.hk/leojia/projects/DeepSR/)], `VideoSR`

**2015 NIPS**

1. **Bidirectional Recurrent Convolutional Networks for Multi-Frame Super-Resolution**, *Huang, Yan; Wang, Wei; Wang, Liang*, [[NIPS](http://papers.nips.cc/paper/5778-bidirectional-recurrent-convolutional-networks-for-multi-frame-super-resolution)], `BRCN`

**2016 CVPR**

1. **Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network**, *Shi, Wenzhe; Caballero, Jose; Huszár, Ferenc; Totz, Johannes; Aitken, Andrew P.; Bishop, Rob; Rueckert, Daniel; Wang, Zehan*, [[arXiv](https://arxiv.org/abs/1609.05158)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2016/html/Shi_Real-time_Single_Image_CVPR_2016_paper.html)], `ESPCN`, `Sub-pixel`

**2016 ICIP**

1. **Super-resolution of compressed videos using convolutional neural networks**, *Kappeler, Armin; Yoo, Seunghwan; Dai, Qiqin; Katsaggelos, Aggelos K.*, [[ICIP](https://ieeexplore.ieee.org/document/7532538)], `CVSRnet`

**2016 TCI**

1. **Video Super-Resolution With Convolutional Neural Networks**, *Kappeler, Armin; Yoo, Seunghwan; Dai, Qiqin; Katsaggelos, Aggelos K.*, [[TCI](https://ieeexplore.ieee.org/document/7444187)], `VSRNet`

**2017 AAAI**

1. **Building an End-to-End Spatial-Temporal Convolutional Network for Video Super-Resolution**, *Guo, Jun; Chao, Hongyang*, [[AAAI](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14733)], `STCN`

**2017 CVPR**

1. **Real-Time Video Super-Resolution with Spatio-Temporal Networks and Motion Compensation**, *Caballero, Jose; Ledig, Christian; Aitken, Andrew; Acosta, Alejandro; Totz, Johannes; Wang, Zehan; Shi, Wenzhe*, [[arXiv](https://arxiv.org/abs/1611.05250)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/html/Caballero_Real-Time_Video_Super-Resolution_CVPR_2017_paper.html)], `VESPCN, STN`

**2017 CVPRW**

1. **FAST: A Framework to Accelerate Super-Resolution Processing on Compressed Videos**, *Zhang, Zhengdong; Sze, Vivienne*, [[arXiv](https://arxiv.org/abs/1603.08968)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Zhang_FAST_A_Framework_CVPR_2017_paper.html)], [[Project](http://www.mit.edu/~sze/fast.html)], `FAST`

**2017 ICCV**

1. **Detail-Revealing Deep Video Super-Resolution**, *Tao, Xin; Gao, Hongyun; Liao, Renjie; Wang, Jue; Jia, Jiaya*, [[arXiv](https://arxiv.org/abs/1704.02738)], [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2017/html/Tao_Detail-Revealing_Deep_Video_ICCV_2017_paper.html)], [[TensorFlow*](https://github.com/jiangsutx/SPMC_VideoSR)], `SPMC`
2. **Robust Video Super-Resolution with Learned Temporal Dynamics**, *Liu, Ding; Wang, Zhaowen; Fan, Yuchen; Liu, Xianming; Wang, Zhangyang; Chang, Shiyu; Huang, Thomas*, [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Robust_Video_Super-Resolution_ICCV_2017_paper.html)], [[Project](http://www.ifp.illinois.edu/~dingliu2/videoSR/)], `Temporal adaptive network`

**2018 arXiv**

1. **Temporally Coherent GANs for Video Super-Resolution (TecoGAN)**, *Chu, Mengyu; Xie, You; Leal-Taixé, Laura; Thuerey, Nils*, [[arXiv](https://arxiv.org/abs/1811.09393)], [[PyTorch*](https://github.com/thunil/TecoGAN)], [[Video](https://ge.in.tum.de/download/2019-TecoGAN/TecoGAN.mp4)], `TecoGAN`

**2018 CVPR**

1. **Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation**, *Jo, Younghyun; Oh, Seoung Wug; Kang, Jaeyeon; Kim, Seon Joo*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.html)], [[TensorFlow*](https://github.com/yhjo09/VSR-DUF)], `VSR-DUF`
2. **Frame-Recurrent Video Super-Resolution**, *Sajjadi, Mehdi S. M.; Vemulapalli, Raviteja; Brown, Matthew*, [[arXiv](https://arxiv.org/abs/1801.04590)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Sajjadi_Frame-Recurrent_Video_Super-Resolution_CVPR_2018_paper.html)], [[TensorFlow*](https://github.com/msmsajjadi/FRVSR)], [[Video](https://vimeo.com/showcase/5053944)], `FRVSR`

**2018 TIP**

1. **Learning Temporal Dynamics for Video Super-Resolution: A Deep Learning Approach**, *Liu, Ding; Wang, Zhaowen; Fan, Yuchen; Liu, Xianming; Wang, Zhangyang; Chang, Shiyu; Wang, Xinchao; Huang, Thomas S.*, [[Project](http://www.ifp.illinois.edu/~dingliu2/videoSR/)], [[TIP](https://ieeexplore.ieee.org/document/8328914)], `Temporal adaptive network`

**2018 TPAMI**

1. **Video Super-Resolution via Bidirectional Recurrent Convolutional Networks**, *Huang, Yan; Wang, Wei; Wang, Liang*, [[TPAMI](https://ieeexplore.ieee.org/document/7919264)], `BRCN`

**2019 CVPR**

1. **Fast Spatio-Temporal Residual Network for Video Super-Resolution**, *Li, Sheng; He, Fengxiang; Du, Bo; Zhang, Lefei; Xu, Yonghao; Tao, Dacheng*, [[arXiv](https://arxiv.org/abs/1904.02870)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Fast_Spatio-Temporal_Residual_Network_for_Video_Super-Resolution_CVPR_2019_paper.html)], `FSTRN`
2. **Recurrent Back-Projection Network for Video Super-Resolution**, *Haris, Muhammad; Shakhnarovich, Greg; Ukita, Norimichi*, [[arXiv](https://arxiv.org/abs/1903.10128)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Haris_Recurrent_Back-Projection_Network_for_Video_Super-Resolution_CVPR_2019_paper.html)], [[Project](https://alterzero.github.io/projects/RBPN.html)], [[PyTorch*](https://github.com/alterzero/RBPN-PyTorch)], `RBPN`

**2019 CVPRW**

1. **Adapting Image Super-Resolution State-Of-The-Arts and Learning Multi-Model Ensemble for Video Super-Resolution**, *Li, Chao; He, Dongliang; Liu, Xiao; Ding, Yukang; Wen, Shilei*, [[arXiv](https://arxiv.org/abs/1905.02462)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Li_Adapting_Image_Super-Resolution_State-Of-The-Arts_and_Learning_Multi-Model_Ensemble_for_Video_CVPRW_2019_paper.html)]
2. **An Empirical Investigation of Efficient Spatio-Temporal Modeling in Video Restoration**, *Fan, Yuchen; Yu, Jiahui; Liu, Ding; Huang, Thomas S*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Fan_An_Empirical_Investigation_of_Efficient_Spatio-Temporal_Modeling_in_Video_Restoration_CVPRW_2019_paper.html)], [[TensorFlow*](https://github.com/ychfan/wdvr_ntire2019)], `WDVR`
3. **EDVR: Video Restoration with Enhanced Deformable Convolutional Networks**, *Wang, Xintao; Chan, Kelvin C. K.; Yu, Ke; Dong, Chao; Loy, Chen Change*, [[arXiv](https://arxiv.org/abs/1905.02716)], [[Project](https://xinntao.github.io/projects/EDVR)], [[PyTorch*](https://github.com/xinntao/EDVR)], `EDVR`
4. **MultiBoot Vsr: Multi-Stage Multi-Reference Bootstrapping for Video Super-Resolution**, *Kalarot, Ratheesh; Porikli, Fatih*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Kalarot_MultiBoot_Vsr_Multi-Stage_Multi-Reference_Bootstrapping_for_Video_Super-Resolution_CVPRW_2019_paper.html)], `MultiBoot`
5. **NTIRE 2019 Challenge on Video Deblurring and Super-Resolution: Dataset and Study**, *Nah, Seungjun; Baik, Sungyong; Hong, Seokil; Moon, Gyeongsik; Son, Sanghyun; Timofte, Radu; Lee, Kyoung Mu*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Nah_NTIRE_2019_Challenge_on_Video_Deblurring_and_Super-Resolution_Dataset_and_CVPRW_2019_paper.html)], `NTIRE`, `REDS`
6. **NTIRE 2019 Challenge on Video Super-Resolution: Methods and Results**, *Nah, Seungjun; Timofte, Radu; Gu, Shuhang; Baik, Sungyong; Hong, Seokil; Moon, Gyeongsik; Son, Sanghyun; Lee, Kyoung Mu*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Nah_NTIRE_2019_Challenge_on_Video_Super-Resolution_Methods_and_Results_CVPRW_2019_paper.html)], `NTIRE`

### 2.4 Domain-specific SR

**2016 ACCV**

1. **Deep Depth Super-Resolution: Learning Depth Super-Resolution Using Deep Convolutional Neural Network**, *Song, Xibin; Dai, Yuchao; Qin, Xueying*, [[ACCV](https://link.springer.com/chapter/10.1007/978-3-319-54190-7_22)]

**2016 ECCV**

1. **ATGV-Net: Accurate Depth Super-Resolution**, *Riegler, Gernot; Rüther, Matthias; Bischof, Horst*, [[arXiv](https://arxiv.org/abs/1607.07988)], [[ECCV](https://link.springer.com/chapter/10.1007/978-3-319-46487-9_17)], `ATGV-Net`
2. **Depth Map Super-Resolution by Deep Multi-Scale Guidance**, *Hui, Tak-Wai; Loy, Chen Change; Tang, Xiaoou*, [[ECCV](https://link.springer.com/chapter/10.1007/978-3-319-46487-9_22)], [[Matlab*](https://github.com/twhui/MSG-Net)], [[Project](http://mmlab.ie.cuhk.edu.hk/projects/guidance_SR_depth.html)], `MSG-Net`

**2017 CVPR**

1. **Perceptual Generative Adversarial Networks for Small Object Detection**, *Li, Jianan; Liang, Xiaodan; Wei, Yunchao; Xu, Tingfa; Feng, Jiashi; Yan, Shuicheng*, [[arXiv](https://arxiv.org/abs/1706.05274)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/html/Li_Perceptual_Generative_Adversarial_CVPR_2017_paper.html)], `Perceptual GAN`

**2017 Pattern Recognition**

1. **Hyperspectral image reconstruction by deep convolutional neural network for classification**, *Li, Yunsong; Xie, Weiying; Li, Huaqing*, [[Pattern Recognition](https://www.sciencedirect.com/science/article/abs/pii/S0031320316303338)], `R-ELM`

**2018 CVPR**

1. **Enhancing the Spatial Resolution of Stereo Images Using a Parallax Prior**, *Jeon, Daniel S.; Baek, Seung-Hwan; Choi, Inchang; Kim, Min H.*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Jeon_Enhancing_the_Spatial_CVPR_2018_paper.html)], [[Project](http://vclab.kaist.ac.kr/cvpr2018/)], [[TensorFlow*](https://github.com/KAIST-VCLAB/stereosr)], `StereoSR`
2. **Feature Super-Resolution: Make Machine See More Clearly**, *Tan, Weimin; Yan, Bo; Bare, Bahetiyaer*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Tan_Feature_Super-Resolution_Make_CVPR_2018_paper.html)], `FSR`
3. **Unsupervised Sparse Dirichlet-Net for Hyperspectral Image Super-Resolution**, *Qu, Ying; Qi, Hairong; Kwan, Chiman*, [[arXiv](https://arxiv.org/abs/1804.05042)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Qu_Unsupervised_Sparse_Dirichlet-Net_CVPR_2018_paper.html)], [[TensorFlow*](https://github.com/aicip/uSDN)], `uSDN`

**2018 NIPS**

1. **Multi-View Silhouette and Depth Decomposition for High Resolution 3D Object Representation**, *Smith, Edward; Fujimoto, Scott; Meger, David*, [[arXiv](https://arxiv.org/abs/1802.09987)], [[NIPS](http://papers.nips.cc/paper/7883-multi-view-silhouette-and-depth-decomposition-for-high-resolution-3d-object-representation)], [[TensorFlow*](https://github.com/EdwardSmith1884/Multi-View-Silhouette-and-Depth-Decomposition-for-High-Resolution-3D-Object-Representation)], `MVD`

**2019 CVPR**

1. **3D Appearance Super-Resolution with Deep Learning**, *Li, Yawei; Tsiminaki, Vagia; Timofte, Radu; Pollefeys, Marc; Van Gool, Luc*, [[arXiv](https://arxiv.org/abs/1906.00925)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_3D_Appearance_Super-Resolution_With_Deep_Learning_CVPR_2019_paper.html)], [[PyTorch*](https://github.com/ofsoundof/3D_Appearance_SR)], `3DASR`
2. **Hyperspectral Image Super-Resolution With Optimized RGB Guidance**, *Fu, Ying; Zhang, Tao; Zheng, Yinqiang; Zhang, Debing; Huang, Hua*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Fu_Hyperspectral_Image_Super-Resolution_With_Optimized_RGB_Guidance_CVPR_2019_paper.html)]
3. **Learning Parallax Attention for Stereo Image Super-Resolution**, *Wang, Longguang; Wang, Yingqian; Liang, Zhengfa; Lin, Zaiping; Yang, Jungang; An, Wei; Guo, Yulan*, [[arXiv](https://arxiv.org/abs/1903.05784)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Learning_Parallax_Attention_for_Stereo_Image_Super-Resolution_CVPR_2019_paper.html)], [[PyTorch*](https://github.com/LongguangWang/PASSRnet)], `PASSRnet`, `Flickr1024`

**2019 TIP**

1. **Channel Splitting Network for Single MR Image Super-Resolution**, *Zhao, Xiaole; Zhang, Yulun; Zhang, Tao; Zou, Xueming*, [[arXiv](https://arxiv.org/abs/1810.06453)], `CSN`



## 3. Unsupervised SR

**2018 CVPR**

1. **"Zero-Shot" Super-Resolution using Deep Internal Learning**, *Shocher, Assaf; Cohen, Nadav; Irani, Michal*, [[arXiv](https://arxiv.org/abs/1712.06087)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.html)], [[Project](http://www.wisdom.weizmann.ac.il/~vision/zssr)], [[TensorFlow*](https://github.com/assafshocher/ZSSR)], `ZSSR`
2. **Deep Image Prior**, *Ulyanov, Dmitry; Vedaldi, Andrea; Lempitsky, Victor*, [[arXiv](https://arxiv.org/abs/1711.10925)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html)], [[Project](https://dmitryulyanov.github.io/deep_image_prior)], [[Python*](https://github.com/DmitryUlyanov/deep-image-prior)], `Deep image prior`

**2018 CVPRW**

1. **Unsupervised Image Super-Resolution Using Cycle-in-Cycle Generative Adversarial Networks**, *Yuan, Yuan; Liu, Siyuan; Zhang, Jiawei; Zhang, Yongbing; Dong, Chao; Lin, Liang*, [[arXiv](https://arxiv.org/abs/1809.00437)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Yuan_Unsupervised_Image_Super-Resolution_CVPR_2018_paper.html)], `CinCGAN`

**2018 ECCV**

1. **To learn image super-resolution, use a GAN to learn how to do image degradation first**, *Bulat, Adrian; Yang, Jing; Tzimiropoulos, Georgios*, [[arXiv](https://arxiv.org/abs/1807.11458)], [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Adrian_Bulat_To_learn_image_ECCV_2018_paper.html)], [[PyTorch*](https://github.com/jingyang2017/Face-and-Image-super-resolution)]



## 4. SR Datasets

|  #   | Dataset      |        #Images         |    #Pixels | Format  | Description                                                  |
| :--: | :----------- | :--------------------: | ---------: | :-----: | :----------------------------------------------------------- |
|  1   | BSDS300      |     300 (200/100)      |    154,401 |   JPG   | Common images, [[ICCV](https://ieeexplore.ieee.org/abstract/document/937655)], [[Project](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)] |
|  2   | Set14        |           14           |    230,203 |   PNG   | Common images, only 14 images, [[Curves and Surfaces](https://link.springer.com/chapter/10.1007/978-3-642-27413-8_47)] |
|  3   | T91          |           91           |     58,853 |   PNG   | Common Images, [[Project](http://www.ifp.illinois.edu/~jyang29/ScSR.htm)], [[TIP](https://ieeexplore.ieee.org/abstract/document/5466111)] |
|  4   | BSDS500      |   500 (200/100/200)    |    154,401 |   JPG   | Common images, [[Project](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)], [[TPAMI](https://ieeexplore.ieee.org/abstract/document/5557884)] |
|  5   | Set5         |           5            |    113,491 |   PNG   | Common images, only 5 images, [[BMVC](http://www.bmva.org/bmvc/2012/BMVC/paper135/index.html)], [[Project](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)] |
|  6   | Urban100     |          100           |    774,314 |   PNG   | Images of real-world structures, [[Matlab*](https://github.com/jbhuang0604/SelfExSR)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2015/html/Huang_Single_Image_Super-Resolution_2015_CVPR_paper.html)] |
|  7   | L20          |           20           | 11,577,492 |   PNG   | Common images, very high-resolution, [[arXiv](https://arxiv.org/abs/1511.02228)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2016/html/Timofte_Seven_Ways_to_CVPR_2016_paper.html)], [[Project](http://www.vision.ee.ethz.ch/~timofter/CVPR2016_ID769_SUPPLEMENTARY/index.html)] |
|  8   | General-100  |          100           |    181,108 |   BMP   | Common images with clear edges but fewer smooth regions, [[arXiv](https://arxiv.org/abs/1608.00367)], [[ECCV](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_25)], [[Project](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)] |
|  9   | Manga109     |          109           |    966,011 |   PNG   | Japanese manga, [[MANPU](https://dl.acm.org/citation.cfm?id=3011551)], [[Project](http://www.manga109.org/en/)] |
|  10  | DIV2K        |   1000 (800/100/100)   |  2,793,250 |   PNG   | Common images, dataset for CVPR competitions (NTIRE), [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Agustsson_NTIRE_2017_Challenge_CVPR_2017_paper.html)], [[Project](https://data.vision.ee.ethz.ch/cvl/DIV2K/)] |
|  11  | WED          |          4744          |    218,664 |   MAT   | Common images, [[Project](https://ece.uwaterloo.ca/~k29ma/exploration/)], [[TIP](https://ieeexplore.ieee.org/document/7752930)] |
|  12  | OutdoorScene |   10624 (10324/300)    |    249,593 |   PNG   | Images of outdoor scenes, [[arXiv](https://arxiv.org/abs/1804.02815)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Recovering_Realistic_Texture_CVPR_2018_paper.html)], [[Project](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN)], [[PyTorch*](https://github.com/xinntao/BasicSR)], [[PyTorch*](https://github.com/xinntao/SFTGAN)] |
|  13  | PIRM         |     200 (100/100)      |    292,021 |   PNG   | Common images, dataset for ECCV competitions (PIRM), [[arXiv](https://arxiv.org/abs/1809.07517)], [[Matlab*](https://github.com/roimehrez/PIRM2018)], [[OpenAccess](http://openaccess.thecvf.com/content_eccv_2018_workshops/w25/html/Blau_2018_PIRM_Challenge_on_Perceptual_Image_Super-resolution_ECCVW_2018_paper.html)], [[Project1](https://pirm.github.io/)], [[Project2](https://www.pirm2018.org/PIRM-SR.html)] |
|  14  | Flickr1024   | 2 * 1024 (800/112/112) |    734,646 |   PNG   | Stereo images, [[arXiv](https://arxiv.org/abs/1903.06332)], [[Project](https://yingqianwang.github.io/Flickr1024/)] |
|  15  | 3DASR        |         3 * 24         |  5,006,868 |   PNG   | 3D textures of 3D objects, [[arXiv](https://arxiv.org/abs/1906.00925)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_3D_Appearance_Super-Resolution_With_Deep_Learning_CVPR_2019_paper.html)], [[PyTorch*](https://github.com/ofsoundof/3D_Appearance_SR)] |
|  16  | City100      |       100 (95/5)       |            |   RAW   | Common images characterizing the R-V degradation under DSLR and smartphone cameras, respectively, [[arXiv](https://arxiv.org/abs/1904.03378)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Camera_Lens_Super-Resolution_CVPR_2019_paper.html)], [[TensorFlow*](https://github.com/ngchc/CameraSR)] |
|  17  | SR-RAW       |  7 * 500 (400/50/50)   |            | JPG/ARW | Raw images produced by real-world computational zoom, [[arXiv](https://arxiv.org/abs/1903.00834)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Image_Super-Resolution_by_Neural_Texture_Transfer_CVPR_2019_paper.html)], [[Project](http://web.eecs.utk.edu/~zzhang61/project_page/SRNTT/SRNTT.html)], [[TensorFlow*](https://github.com/ZZUTK/SRNTT)] |
|  18  | CUFED5       |          756           |    174,151 |   PNG   | Reference images, each image group consists of 1 root image and 4 reference images at different similarity level, [[arXiv](https://arxiv.org/abs/1905.05169)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Zoom_to_Learn_Learn_to_Zoom_CVPR_2019_paper.html)], [[Project1](https://ceciliavision.github.io/project-pages/project-zoom.html)], [[Project2](http://vladlen.info/publications/zoom-learn-learn-zoom/)], [[TensorFlow*](https://github.com/ceciliavision/zoom-learn-zoom)], [[Video](https://www.youtube.com/watch?v=if6hZKglgL0)] |

- "#Images" represents the total number of images in the dataset, where images generated manually are excluded (e.g., LR images obtained by bicubic down-sampling on a HR image).
- "#Pixels" represents the average number of pixels in all the images in the dataset. Since the resolution of the images tends to be different in the dataset, this value can better represent the size of the image in the dataset. 
- At present, we mainly include image super-resolution datasets, and other datasets (such as face image SR, video SR) will be supplemented later.

**Corresponding Papers**

1. **A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics**, *Martin, David; Fowlkes, Charless; Tal, Doron; Malik, Jitendra*, **ICCV 2001**, [[ICCV](https://ieeexplore.ieee.org/abstract/document/937655)], [[Project](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)], `BSDS300`
2. **On Single Image Scale-Up Using Sparse-Representations**, *Zeyde, Roman; Elad, Michael; Protter, Matan*, **Curves and Surfaces 2010**, [[Curves and Surfaces](https://link.springer.com/chapter/10.1007/978-3-642-27413-8_47)], `Set14`
3. **Image Super-Resolution Via Sparse Representation**, *Yang, Jianchao; John, Wright; Thomas, Huang; Ma, Yi*, **TIP 2010**, [[Project](http://www.ifp.illinois.edu/~jyang29/ScSR.htm)], [[TIP](https://ieeexplore.ieee.org/abstract/document/5466111)], `T91`
4. **Contour Detection and Hierarchical Image Segmentation**, *Arbeláez, Pablo; Maire, Michael; Fowlkes, Charless; Malik, Jitendra*, **TPAMI 2011**, [[Project](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)], [[TPAMI](https://ieeexplore.ieee.org/abstract/document/5557884)], `BSDS500`
5. **Low-Complexity Single-Image Super-Resolution based on Nonnegative Neighbor Embedding**, *Bevilacqua, Marco; Roumy, Aline; Guillemot, Christine; Morel, Marie-line Alberi*, **BMVC 2012**, [[BMVC](http://www.bmva.org/bmvc/2012/BMVC/paper135/index.html)], [[Project](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)], `Set5`
6. **Single image super-resolution from transformed self-exemplars**, *Huang, Jia-Bin; Singh, Abhishek; Ahuja, Narendra*, **CVPR 2015**, [[Matlab*](https://github.com/jbhuang0604/SelfExSR)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2015/html/Huang_Single_Image_Super-Resolution_2015_CVPR_paper.html)], `SelfExSR`, `Urban100`
7. **Seven ways to improve example-based single image super resolution**, *Timofte, Radu; Rothe, Rasmus; Van Gool, Luc*, **CVPR 2016**, [[arXiv](https://arxiv.org/abs/1511.02228)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2016/html/Timofte_Seven_Ways_to_CVPR_2016_paper.html)], [[Project](http://www.vision.ee.ethz.ch/~timofter/CVPR2016_ID769_SUPPLEMENTARY/index.html)], `IA`, `L20`
8. **Accelerating the Super-Resolution Convolutional Neural Network**, *Dong, Chao; Loy, Chen Change; Tang, Xiaoou*, **ECCV 2016**, [[arXiv](https://arxiv.org/abs/1608.00367)], [[ECCV](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_25)], [[Project](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)], `FSRCNN`, `General-100`
9. **Manga109 dataset and creation of metadata**, *Fujimoto, Azuma; Ogawa, Toru; Yamamoto, Kazuyoshi; Matsui, Yusuke; Yamasaki, Toshihiko; Aizawa, Kiyoharu*, **MANPU 2016**, [[MANPU](https://dl.acm.org/citation.cfm?id=3011551)], [[Project](http://www.manga109.org/en/)], `Manga109`
10. **NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study**, *Agustsson, Eirikur; Timofte, Radu*, **CVPRW 2017**, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Agustsson_NTIRE_2017_Challenge_CVPR_2017_paper.html)], [[Project](https://data.vision.ee.ethz.ch/cvl/DIV2K/)], `NTIRE`, `DIV2K`
11. **Waterloo Exploration Database: New Challenges for Image Quality Assessment Models**, *Ma, Kede; Duanmu, Zhengfang; Wu, Qingbo; Wang, Zhou; Yong, Hongwei; Li, Hongliang; Zhang, Lei*, **TIP 2017**, [[Project](https://ece.uwaterloo.ca/~k29ma/exploration/)], [[TIP](https://ieeexplore.ieee.org/document/7752930)], `WED`
12. **Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform**, *Wang, Xintao; Yu, Ke; Dong, Chao; Loy, Chen Change*, **CVPR 2018**, [[arXiv](https://arxiv.org/abs/1804.02815)], [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Recovering_Realistic_Texture_CVPR_2018_paper.html)], [[Project](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN)], [[PyTorch*](https://github.com/xinntao/BasicSR)], [[PyTorch*](https://github.com/xinntao/SFTGAN)], `SFT-GAN`, `OutdoorScene`
13. **The 2018 PIRM Challenge on Perceptual Image Super-resolution**, *Blau, Yochai; Mechrez, Roey; Timofte, Radu; Michaeli, Tomer; Zelnik-Manor, Lihi*, **ECCVW 2018**, [[arXiv](https://arxiv.org/abs/1809.07517)], [[Matlab*](https://github.com/roimehrez/PIRM2018)], [[OpenAccess](http://openaccess.thecvf.com/content_eccv_2018_workshops/w25/html/Blau_2018_PIRM_Challenge_on_Perceptual_Image_Super-resolution_ECCVW_2018_paper.html)], [[Project1](https://pirm.github.io/)], [[Project2](https://www.pirm2018.org/PIRM-SR.html)], `PIRM`
14. **Flickr1024: A Dataset for Stereo Image Super-Resolution**, *Wang, Yingqian; Wang, Longguang; Yang, Jungang; An, Wei; Guo, Yulan*, **arXiv 2019**, [[arXiv](https://arxiv.org/abs/1903.06332)], [[Project](https://yingqianwang.github.io/Flickr1024/)], `Flickr1024`
15. **3D Appearance Super-Resolution with Deep Learning**, *Li, Yawei; Tsiminaki, Vagia; Timofte, Radu; Pollefeys, Marc; Van Gool, Luc*, **CVPR 2019**, [[arXiv](https://arxiv.org/abs/1906.00925)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_3D_Appearance_Super-Resolution_With_Deep_Learning_CVPR_2019_paper.html)], [[PyTorch*](https://github.com/ofsoundof/3D_Appearance_SR)], `3DASR`
16. **Camera Lens Super-Resolution**, *Chen, Chang; Xiong, Zhiwei; Tian, Xinmei; Zha, Zheng-Jun; Wu, Feng*, **CVPR 2019**, [[arXiv](https://arxiv.org/abs/1904.03378)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Camera_Lens_Super-Resolution_CVPR_2019_paper.html)], [[TensorFlow*](https://github.com/ngchc/CameraSR)], `CameraSR`, `City100`
17. **Image Super-Resolution by Neural Texture Transfer**, *Zhang, Zhifei; Wang, Zhaowen; Lin, Zhe; Qi, Hairong*, **CVPR 2019**, [[arXiv](https://arxiv.org/abs/1903.00834)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Image_Super-Resolution_by_Neural_Texture_Transfer_CVPR_2019_paper.html)], [[Project](http://web.eecs.utk.edu/~zzhang61/project_page/SRNTT/SRNTT.html)], [[TensorFlow*](https://github.com/ZZUTK/SRNTT)], `SRNTT`, `CUFED5`
18. **Zoom to Learn, Learn to Zoom**, *Zhang, Xuaner; Chen, Qifeng; Ng, Ren; Koltun, Vladlen*, **CVPR 2019**, [[arXiv](https://arxiv.org/abs/1905.05169)], [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Zoom_to_Learn_Learn_to_Zoom_CVPR_2019_paper.html)], [[Project1](https://ceciliavision.github.io/project-pages/project-zoom.html)], [[Project2](http://vladlen.info/publications/zoom-learn-learn-zoom/)], [[TensorFlow*](https://github.com/ceciliavision/zoom-learn-zoom)], [[Video](https://www.youtube.com/watch?v=if6hZKglgL0)], `SR-RAW`, `CoBi`
19. **NTIRE 2019 Challenge on Video Deblurring and Super-Resolution: Dataset and Study**, *Nah, Seungjun; Baik, Sungyong; Hong, Seokil; Moon, Gyeongsik; Son, Sanghyun; Timofte, Radu; Lee, Kyoung Mu*, **CVPRW 2019**, [[OpenAccess](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Nah_NTIRE_2019_Challenge_on_Video_Deblurring_and_Super-Resolution_Dataset_and_CVPRW_2019_paper.html)], `NTIRE`, `REDS`



## 5. SR Metrics

| Metric | Papers |
| --- | --- |
| MS-SSIM | **Multiscale structural similarity for image quality assessment**, *Wang, Zhou; Simoncelli, Eero P.; Bovik, Alan C.*, **ACSSC 2003**, [[ACSSC](https://ieeexplore.ieee.org/document/1292216)], `MS-SSIM` |
| SSIM | **Image Quality Assessment: From Error Visibility to Structural Similarity**, *Wang, Zhou; Bovik, Alan C.; Sheikh, Hamid R.; Simoncelli, Eero P*, **TIP 2004**, [[TIP](https://ieeexplore.ieee.org/document/1284395)], `SSIM` |
| IFC | **An information fidelity criterion for image quality assessment using natural scene statistics**, *Sheikh, Hamid Rahim; Bovik, Alan Conrad; de Veciana, Gustavo de Veciana*, **TIP 2005**, [[TIP](https://ieeexplore.ieee.org/document/1532311/)], `IFC` |
| VIF | **Image information and visual quality**, *Sheikh, Hamid Rahim; Bovik, Alan C.*, **TIP 2006**, [[TIP](https://ieeexplore.ieee.org/document/1576816)], `VIF` |
| FSIM | **FSIM: A Feature Similarity Index for Image Quality Assessment**, *Zhang, Lin; Zhang, Lei; Mou, Xuanqin; Zhang, David*, **TIP 2011**, [[Project](http://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/FSIM.htm)], [[TIP](https://ieeexplore.ieee.org/document/5705575)], `FSIM` |
| NIQE | **Making a “Completely Blind” Image Quality Analyzer**, *Mittal, Anish; Soundararajan, Rajiv; Bovik, Alan C.*, **Signal Processing Letters 2013**, [[Matlab*](https://github.com/csjunxu/Bovik_NIQE_SPL2013)], [[Signal Processing Letters](https://ieeexplore.ieee.org/document/6353522)], `NIQE` |
| Ma | **Learning a no-reference quality metric for single-image super-resolution**, *Ma, Chao; Yang, Chih-Yuan; Yang, Xiaokang; Yang, Ming-Hsuan*, **CVIU 2017**, [[arXiv](https://arxiv.org/abs/1612.05890)], [[CVIU](https://www.sciencedirect.com/science/article/pii/S107731421630203X)], [[Matlab*](https://github.com/chaoma99/sr-metric)], [[Project](https://sites.google.com/site/chaoma99/sr-metric)], `Ma` |



## 6. Survey Resources

* **Taxonomy of our survey**

![](resources/survey_taxonomy.jpg)

* **SR Benchmarking**

![](resources/survey_sr_benchmark.jpg)



## 7. Other Resources

- [Papers With Code : Super Resolution](https://paperswithcode.com/task/super-resolution)
- [YapengTian/Single-Image-Super-Resolution](https://github.com/YapengTian/Single-Image-Super-Resolution)
- [LoSealL/VideoSuperResolution](https://github.com/LoSealL/VideoSuperResolution)



<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" /></a>