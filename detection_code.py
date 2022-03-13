import cv2
import numpy as np


cap = cv2.VideoCapture("") # işlenecek olan videonun adresini girip videoyu okuyoruz

while True: # videoda her bir frame'in en ve boy bilgisini almalıyız
    ret, frame = cap.read() #frame okuma

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    frame_blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False) # her bir frame'i blob formata dönüştürüyoruz

    labels = ["PLAKA"] # etiketlerimiz

    color = (0,255,0) # bulunan etiketin rengi
    color = np.tile(color, (18, 1))

    model = cv2.dnn.readNetFromDarknet("","") # nesne tespitinde kullanılacak weight ve cfg dosyasının adresini alıyoruz
    layers = model.getLayerNames()
    output_layer = [layers[layer - 1] for layer in model.getUnconnectedOutLayers()] # çıktı katmanlarını buluyoruz

    model.setInput(frame_blob) # blob yaptıgımız tensörü modelimize veriyoruz .
    detection_layers = model.forward(output_layer) # ve tespit edilmiş layerlerımızı alyoruz

    for detection_layers in detection_layers:
        for object_detection in detection_layers:
            scores = object_detection[5:] # ilk 5 deger bouding box ile ilgili biz bundan sonrakileri alıcaz
            predicted_id = np.argmax(scores) # scores tuttugu degerler icinden max indexi bulacagız
            confidence = scores[predicted_id]# güven skoruna erişecegiz
            if confidence > 0.30:  #nesne tespitinde yuzde kac basarıda nesnenin plaka olarak belirtilmesi gerektigini belirliyoruz
                label = labels[0]   # bir tane etiketimiz oldugu icin direkt labels[0] dedik
                bounding_box = object_detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])  # en ve boy ile bounding_box ı genişlettik
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int") # bounding box ta gelen degeri int e cevirecegiz bunlar float cunku

                start_x = int(box_center_x - (box_width / 2)) # diktörtgenimizin baslangıc ve bitiş noktalarını belirledik
                start_y = int(box_center_y - (box_height / 2))

                end_x = start_x + box_width # baslangıca boyutu eklersek bitişni bulmus oluruz
                end_y = start_y + box_height

                box_color = color[0] # yukarıda label ımız icin renk belirlemiştik
                box_color = [int(each) for each in box_color]

                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 1)
                cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        cv2.imshow("detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): # q ya basınca video kapansın
            break

cap.release()# tüm pencereleri kapatmak icin
cv2.destroyAllWindows()