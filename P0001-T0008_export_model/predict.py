import paddlex as pdx
import cv2

print("Loading model...")
model = pdx.load_model('./inference_model/inference_model/')
print("Model loaded.")


cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

flag = cap.isOpened()
while (flag):
    ret, im = cap.read()
    #im = im.astype('float32')
    # ret, frame1 = cap1.read()
    if not ret:
        print('打开失败')
        break
    result = model.predict(im)


    # 输出分类结果
    if model.model_type == "classifier":
        print(result)

    # 可视化结果, 对于检测、实例分割务进行可视化
    if model.model_type == "detector":
        # threshold用于过滤低置信度目标框
        # 可视化结果保存在当前目录
        pdx.det.visualize(im, result, threshold=0.5, save_dir=None)

    # 可视化结果, 对于语义分割务进行可视化
    if model.model_type == "segmenter":
        # weight用于调整结果叠加时原图的权重
        # 可视化结果保存在当前目录
        pdx.seg.visualize(im, result, weight=0.0, save_dir='./')

    cv2.imshow('my_goals', im)
    k = cv2.waitKey(10) & 0xFF
    if  k == ord('q'):
        break
cap.release()
# cap1.release()
cv2.destroyAllWindows()

