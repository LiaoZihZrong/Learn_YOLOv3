#  YOLOv3練習(v1-v4統整資料)
> * 主題: keras-yolo3-qwe練習
> * 日期: 2020/10/18
> * 備註: 本次參考資料較多，請務必詳細閱讀
> 
###### tags: `CV`

## 目錄
[TOC]

---
# 課程檔案
> https://1drv.ms/u/s!AqstO-BYCWeDw3cz-5aURIFie6Ml?e=Ukqsrr

---

# 老師提供的三項教學:
#### 物件偵測基本原理: 
object detection
> * https://zhuanlan.zhihu.com/p/21412911

#### YOLOv3
> * 1.zhihu: https://zhuanlan.zhihu.com/p/37850811
> * 2.practical: https://www.analyticsvidhya.com/blog/2018/12/practical-guide-object-detection-yolo-framewor-python/

---
 
# 我的學習路徑(含參考資料):
## 1. 請先觀看以下影片-簡介 YOLOv3
* YOLOv3簡介(影片) https://youtu.be/2_ga_k9ZpF4
* YOLOv3改進(影片) https://youtu.be/xGfpd5HpQ8k

## 2. 簡介 YOLOv3
> ==本處為大略式的介紹，節錄以下參考資料(非常重要)==
* YOLOv3 演進
https://medium.com/@chingi071/yolo%E6%BC%94%E9%80%B2-2-85ee99d114a1

* 圖像解說 YOLO v3
https://luckmoonlight.github.io/2018/11/28/yoloV1yolov2yoloV3

* 基於深度學習的物件偵測(含YoloV3)
https://mropengate.blogspot.com/2018/06/yolo-yolov3.html

* YOLOv3簡介(圖片較多)
https://github.com/LiaoZihZrong/learn_YOLOv3

### 2-1 YOLOv3 論文
> YOLOv3 論文下載: 
https://1drv.ms/b/s!AqstO-BYCWeDw3ttHma2-aPxP0EQ?e=71GhG0


### 2-2 改良後的YOLOv3:
#### (1) YOLOv3 在 YOLOv2 的基礎上，改用了 Darknet-53


![](https://i.imgur.com/oGRN8mB.jpg)

![](https://i.imgur.com/O0wunfI.png)

#### (2)利用多尺度特徵圖 (feature map) 進行檢測

> YOLOv3 借鑒了 FPN 的方法，採用多尺度的 feature map 對不同大小的物體進行檢測，提升小物體預測能力

![](https://i.imgur.com/nIXxW2p.png)
![](https://i.imgur.com/VNWceK4.png)


> YOLOv3 通過下採樣32倍、16倍、8倍得到3個不同尺度的 feature map
例如輸入416x416的圖片
則會得到13x13 (416/32)、26x26 (416/16)、52x52 (416/8)

![](https://i.imgur.com/FMgQCyP.png)


> 13x13 feature map (最大的感受野) 用於偵測大物體，所以用較大的Anchor prior (116x90), (156x198), (373x326)
26x26 feature map (中等的感受野) 用於偵測中等大小的物體，所以用中等的Anchor prior (30x61), (62x45), (59x119)
52x52eature map (較小的感受野) 用於檢測小物體，所以用較小的Anchor prior (10x13), (16x30), (33x23)
#### (3)Anchor box
> 每個尺度的 feature map 會預測出3個 Anchor prior
而 Anchor prior 的大小則採用K-means進行聚類分析 
(YOLOv3 延續了 YOLOv2 的作法)。

> 在COCO資料集上，按照輸入圖片的尺寸為416x416
得到9種聚類結果 (Anchor prior的wxh)：
(10x13), (16x30), (33x23),
(30x61),(62x45), (59x119), 
(116x90), (156x198), (373x326)



#### (4)改用多個獨立的 Logistic regression 分類器取代softmax 

> Class Prediction-使用 binary cross-entropy
YOLO 之前都是使用 softmax 去分類每個 bndBox，
而預測目標裡可能有重疊的標籤 
因此 YOLOv3 改採用多個獨立的 Logistic regression 分類器


#### (5) IOU
> Bounding Box Prediction
YOLOv3 使用 Logistic regression 來預測每個 bndBox 的 confidence，以 bndBox 與 ground truth 的 IOU 為判定標準，對每個 ground truth 只分配一個最好的bndBox。利用這個方式，在做 Detect 之前可以減少不必要的 Anchor 並檢少計算量

> * 正例: 將 IOU最高的bndBox ，confidence score 設為1
> * 忽略樣例: 其他不是最高 IOU 的 bndBox 並且 IOU 大於閾值 (threshold，預測為0.5) ，則忽略這些 bndBox，不計算 loss
> * 負例: 若 bndBox 沒有與任一 ground truth 對應，則減少其 confidence score

#### 補充:邊框預測(整合上述1~5)
* 請先閱讀參考資料↓:
==超詳細的Yolov3邊框預測分析== https://zhuanlan.zhihu.com/p/49995236

> 遵循YOLO9000，我們的系統使用錨定框來預測邊界框。 
網絡為每個邊界框tx，ty，tw，th預測4個坐標。 
如果單元格從圖像的左上角偏移了（cx，cy）並且先驗邊界框的寬度和高度為pw，ph，則預測對應於：

$b_x=σ(t_x)+c_x$
$b_y=σ(t_y)+c_y$
$b_w=p_we^{t_w}$
$b_h=p_he^{t_h}$
$P_r(object)*IOU(b,object)=σ(t_o)$

![](https://i.imgur.com/UXjPxFV.png)

補充:
> (tx,ty):目標中心點相對於該點所在網格左上角的偏移量，經過sigmoid,即值為 [ 0, 1 ]
(cx,cy):該點所在網格的左上角距離最左上角相差的格子數
(pw,ph):anchor box 的邊常
(tw,th):預測邊框的寬和高
最終得到的邊框坐標值是bx,by,bw,bh.而網絡學習目標是tx,ty,tw,th





#### (6) YOLOv3 所使用之Loss
![](https://i.imgur.com/g1uwtkZ.jpg)



### 速度與準確率
> 若採用COCO mAP50 做評估標準 (不介意 bndBox 預測的準確性)
YOLOv3 的表現達到57.9%，與 RetinaNet 的結果相近，並且速度快4 倍

![](https://i.imgur.com/hSmv2vG.png)


## YOLOv1-v3比較(包含YOLOv4採取的技術)
> ==參考以下資料整合(非常重要)==
> * YOLO v1、v2、v3 簡介-承軒學長 
https://hackmd.io/@ZZ/Hyut_fOwP
> * YOLO v1~v4 簡介
https://www.cnblogs.com/wujianming-110117/p/12840766.html
> * YOLO v1~3 比較
https://blog.cavedu.com/2019/07/25/yolo-identification-model/
> * YOLO v1~v4
https://tangh.github.io/articles/yolo-from-v1-to-v4/
> * YOLO v4 介紹
https://medium.com/@chingi071/yolo%E6%BC%94%E9%80%B2-3-yolov4%E8%A9%B3%E7%B4%B0%E4%BB%8B%E7%B4%B9-5ab2490754ef
> * IoU/GIoU/DIoU/CIoU Loss簡介 
https://zhuanlan.zhihu.com/p/104236411
> * YOLO v1-v5簡介
https://zhuanlan.zhihu.com/p/136382095

#### YOLOv1、v2、v3比較
![](https://i.imgur.com/TWw0RqW.jpg)
![](https://i.imgur.com/h2bOkpt.jpg)
![](https://i.imgur.com/7I3IMy4.jpg)

---

## YOLOv4簡介 
:::info
:bulb: 請參考上方引用資料，本篇不詳細討論YOLOv4，僅條列式節錄
:::
#### YOLOv4論文下載:
> https://1drv.ms/b/s!AqstO-BYCWeDw3y_KIX9j6oo99qR?e=d2vvEh

#### YOLOv4的五個基本組件：
> CBM：Yolov4網絡結構中的最小組件，由Conv+Bn+Mish激活函數三者組成。
> CBL：由Conv+Bn+Leaky_relu激活函數三者組成。
> Res unit：借鑑Resnet網絡中的殘差結構，讓網絡可以構建的更深。
> CSPX：借鑑CSPNet網絡結構，由三個卷積層和X個Res unint模塊Concate組成。
> SPP：採用1×1，5×5，9×9，13×13的最大池化的方式，進行多尺度融合。




![](https://i.imgur.com/Fo9HOy2.jpg)
#### YOLOv4之四部分:
> 輸入端：主要包括Mosaic數據增強、cmBN、SAT自對抗訓練
> BackBone主幹網絡：CSPDarknet53、Mish激活函數、Dropblock
> Neck：目標檢測網絡在BackBone和最後的輸出層之間往往會插入一些層，如:SPP模塊、FPN+PAN結構
> Head: Base on YOLOv3
> Prediction：輸出層的錨框機制和Yolov3相同，主要改進的是訓練時的損失函數CIOU_Loss，以及預測框篩選的nms變為DIOU_nms

> * ==補充==:
> Bounding Box Regeression的Loss發展過程：
Smooth L1 Loss-> IoU Loss（2016）-> GIoU Loss（2019）-> DIoU Loss（2020）->CIoU Loss（2020）


![](https://i.imgur.com/U9KHe8R.jpg)

![](https://i.imgur.com/hKWvZgc.jpg)

#### YOLOv3、v4比較
![](https://i.imgur.com/p8q6WeG.jpg)
#### YOLOv1、v2、v3、v4比較
![](https://i.imgur.com/NWEGvZG.jpg)
![](https://i.imgur.com/KUVLgzH.png)

:::info
:bulb: 引述參考資料，YOLOv4選用新技術，並有效提升其辨識效果，相當推薦學習!
:::
---

# 實作指令

在命令列模式下，切到 keras-yolo3-qwe 目錄
如要偵測 pedestrian.jpg 可執行下面指令
```
$ python yolo_video.py --input pedestrian.jpg
```
或
```
$ python yolo_video.py --input pedestrian.jpg --output pedestrian_pred.jpg
```
上一行會將偵測結果存到 pedestrian_pred.jpg

#### 執行結果:
> ![image](https://cdkqfg.dm.files.1drv.com/y4m0__I8l421X8Kl6QJ85pw8YPVJqmbvVHjmYV5cbTHhkhbGEWNFf_idgrSE5k-6Ae53V5cH268lGt3tT32I493BFHdYtQb_oZ3xslKq92UevE0TzCD_shgTW4HyhHNqo17IMwS4zJO7U1ssV1Ew-CSQkhcUF0FwKgJfYc9L4RfkIoW9I4r3AdMhO5o9HqWKOsgWlvCR5qKHtDk8jy4EVyxpA?width=770&height=513&cropmode=none)


如要偵測 people.avi 可執行下面指令
```
$ python yolo_video.py --input people.avi
```
或
```
$ python yolo_video.py --input people.avi --output people_pred.avi
```

上一行會將偵測結果存到 people_pred.avi

#### 執行結果:

 {%youtube UfozZfL5lr4 %}
 
---
# 題目:
### 了解YOLOv3程式碼(1)-detect_image
#### 參考資料
> yolov3 keras版本yolo.py函式解析 https://blog.csdn.net/qq_43211132/article/details/102988139
#### <參考運作流程>
def detect_image(self, image):（重要）
開始計時->
①調用letterbox_image函數，即：先生成一個用“絕對灰”R128-G128-B128填充的416×416新圖片，然後用按比例縮放（採樣方式：BICUBIC）後的輸入圖片粘貼，粘貼不到的部分保留為灰色。
②model_image_size定義的寬和高必須是32的倍數；若沒有定義model_image_size，將輸入的尺寸調整為32的倍數，並調用letterbox_image函數進行縮放。
③將縮放後的圖片數值除以255，做歸一化。
④將（416,416,3）數組調整為（1,416,416,3），滿足網絡輸入的張量格式：image_data。

->
①運行self.sess.run（）輸入參數：輸入圖片416×416，學習模式0測試/1訓練。 self.yolo_model.input: image_data，self.input_image_shape: [image.size[1], image.size[0]]，K.learning_phase(): 0。
②self.generate（），讀取：model路徑、anchor box、coco類別、加載模型yolo.h5.，對於80中coco目標，確定每一種目標框的繪製顏色，即：將（x/80,1.0 ,1.0）的顏色轉換為RGB格式，並隨機調整顏色一遍肉眼識別，其中：一個1.0表示飽和度，一個1.0表示亮度。


->
①yolo_eval(self.yolo_model.output),max_boxes=20,
每張圖沒類最多檢測20個框。
②將anchor_box分為3組，分別分配給三個尺度，
yolo_model輸出的feature map
③特徵圖越小，感受野越大，對大目標越敏感，
選大的anchor box->分別對三個feature map運行
out_boxes, out_scores, out_classes，
返回boxes、scores、classes。


:::info
:bulb: yolo.py 內的 detect_image( )
:::
:::spoiler Code
```python=
#請參考註解

#def detect_image(self, image):
#detect_image作用：
    #要求圖片尺寸是32的倍數
    #原因:執行的是5次step為2的捲積操作
    #即圖片的尺寸是416*416，因在最底層中的特徵圖大小是13*13 (13*32=416)
def detect_image(self, image, output_path=""):
    start = timer()  #計時器
    # 調用letterbox_image()函數
    # 生成一個用 絕對灰 R128-G128-B128 填充的 (416x416)新圖片
    # 然後用按比例縮放（BICUBIC）後的輸入圖片黏貼，黏貼不到的部分保留為灰色
    if self.model_image_size != (None, None):
        assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
        # assert語法格式 model_image_size[0][1]指圖像的w和h，且必須是32的整數倍
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        # letterbox_image調整尺寸為(w,h)
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    print(image_data.shape)
    #（416，416,3）
    
    image_data /= 255.
    #除以255 (正規化)
    
    # batch dimension
    # 設定一维 -> (1,416,416,3) 格式(bitch, w, h, c)
    image_data = np.expand_dims(image_data, 0) 
 
    #求boxes,scores,classes
    #請參照程式上方 generate()
    out_boxes, out_scores, out_classes = self.sess.run(
        [self.boxes, self.scores, self.classes],
        feed_dict={
            #參數設定
            #圖像數據
            self.yolo_model.input: image_data,
            #尺寸416x416
            self.input_image_shape: [image.size[1], image.size[0]],
            #模式 0：測試模型  1：訓練模型
            K.learning_phase(): 0
        })

    #使用Pillow繪圖庫 繪製邊框、邊框寬度、文字
    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
    #設定字體
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    #設定框線寬度、厚度
    thickness = (image.size[0] + image.size[1]) // 300
    
    #使用Pillow繪圖庫 對C個目標類別中的每個目標框i 處理
    for i, c in reversed(list(enumerate(out_classes))):
        #目標類別的名字
        predicted_class = self.class_names[c]
        #框
        box = out_boxes[i]
        #框信度
        score = out_scores[i]
        #標籤
        label = '{} {:.2f}'.format(predicted_class, score)
        #繪製輸入的原始圖片
        draw = ImageDraw.Draw(image)
        #標籤文字 -> label的寬與高（pixels）
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        
        #目標框的上、左  兩個座標四捨五入取小數後一位
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        
        #目標框的下、右  兩個座標四捨五入取小數後一位
        #與圖片的尺寸相比取最小值
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))
        #邊框 確定標籤起始點位置:左、下
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # 繪製目標框，線條寬度為thickness
        for i in range(thickness):
        #畫框
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=self.colors[c])
        draw.rectangle(
        #文字.背景
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=self.colors[c])
            
        # 標籤內容
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
        
    # 計時器結束
    end = timer()
    #顯示時長
    print(end - start)
    
    isOutput = True if output_path != "" else False
    if isOutput:
      image.save(output_path)
    return image
    
def close_session(self):
    self.sess.close()

```
:::

↑(程式碼-請點開上方)

---

### 了解YOLOv3程式碼(2)-detect_video

:::info
:bulb: yolo.py 內的 detect_video( )
:::

:::spoiler Code
```python=
#請參考註解
#利用 cv2 將影片變成一幀一幀
#並使用 detect_image 辨識後顯示結果
def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    #開啟影片
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    #讀取影片資訊
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    #計時器開始
    prev_time = timer()
    
    #辨識開始
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        
        #使用detect_image函式變式圖片 
        #(請參照上方detect_image程式解析)
        image = yolo.detect_image(image)
        #將結果存至result
        result = np.asarray(image)
        curr_time = timer()
        
        #計算 處理時間
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        #計算FPS
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
            
        #顯示辨識結果
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        #結束結果畫面
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()
```
:::

↑(程式碼-請點開上方)


* 程式各函式運作流程圖:
https://1drv.ms/u/s!AqstO-BYCWeDxBD-DKTXrqyxvCdu




---

### 程式碼與題目 ### 
> https://1drv.ms/u/s!AqstO-BYCWeDw3cz-5aURIFie6Ml?e=Ukqsrr


:::info
備註:本篇主要為統整閱讀完的資料，感謝上述引用資料的分享!
:::