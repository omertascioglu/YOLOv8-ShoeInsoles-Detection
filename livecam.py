from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from ultralytics import YOLO
import numpy as np
import argparse
import imutils
import cv2
import math


class point:
    x = 0
    y = 0

plL = point()
plR = point()
plU = point()
plD = point()

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

cam = cv2.VideoCapture(0)

while True:
    _, image = cam.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
              (255, 0, 255))
    refObj = None
    pixelsPerMetric = None

    model = YOLO("best.pt")

    results = model(image, conf=0.75)

    for r in results:
        data = r.boxes
        for box in data:
            tl, tr, bl, br = box.xyxy[0]
            tl, tr, bl, br = int(tl), int(tr), int(bl), int(br)

            cv2.rectangle(image, (tl, tr), (bl, br), (243, 239, 224), 3)

            org = [tl, tr]
            font = cv2.FONT_HERSHEY_COMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(image, """Ayakkabi Tabani""", org, font, fontScale, color, thickness)

    for c in contours:
        if cv2.contourArea(c) < 100:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)

        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])

        if refObj is None:
            (tl, tr, br, bl) = box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            refObj = (box, (cX, cY), D / 2.8)
            continue


        orig = image.copy()
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

        refCoords = np.vstack([refObj[0], refObj[1]])
        objCoords = np.vstack([box, (cX, cY)])

        plL.x = box[0][0]
        plL.y = box[0][1]

        plR.x = box[1][0]
        plR.y = box[1][1]

        plU.x = box[2][0]
        plU.y = box[2][1]

        plD.x = box[3][0]
        plD.y = box[3][1]

        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        if pixelsPerMetric is None:
            pixelsPerMetric = 27.6

        dimA = dA / (pixelsPerMetric * 4)
        dimB = dB / pixelsPerMetric

        cv2.putText(orig, "{:.1f}cm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}cm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

        rp1 = point()
        rp2 = point()

        Angle = 0

        if (dA >= dB):
            rp1.x = tltrX
            rp1.y = tltrY
            rp2.x = blbrX
            rp2.y = blbrY
        else:
            rp1.x = tlblX
            rp1.y = tlblY
            rp2.x = trbrX
            rp2.y = trbrY

        delX = (rp2.x - rp1.x) / (math.sqrt(((rp2.x - rp1.x) ** 2) + ((rp2.y - rp1.y) ** 2)))
        delY = (rp2.y - rp1.y) / (math.sqrt(((rp2.x - rp1.x) ** 2) + ((rp2.y - rp1.y) ** 2)))

        cv2.line(orig, (int(rp1.x - delX * 350), int(rp1.y - delY * 350)),
                 (int(rp2.x + delX * 250), int(rp2.y + delY * 250)), (205, 0, 0), 2)

        x, y, z = image.shape

        cv2.line(orig, (0, int(y / 3)), (x * 20, int(y / 3)), (0, 0, 0), 2)

        if rp2.x - rp1.x != 0:
            gradient = (rp2.y - rp1.y) / (rp2.x - rp1.x)
            Angle = math.atan(gradient)
            Angle = Angle * 57.2958

            if Angle < 0:
                Angle = Angle + 180
        else:
            Angle = 90

        cv2.putText(orig, "{:.4f}".format(Angle) + " Degrees",
                    (330, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)


        for ((xB, yB), color) in zip(objCoords, colors):
            cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)

        cv2.imshow("Image", orig)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
