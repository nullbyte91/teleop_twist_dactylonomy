#!/usr/bin/env python3

import cv2
import time
import mediapipe as mp
from argparse import ArgumentParser
import rospy
from geometry_msgs.msg import Twist

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", default=0, required= False, type=int,
                        help="video stream number")
    parser.add_argument("-p", "--publisher", default="/cmd_vel", required= False, type=str,
                help="Twist publisher namespace")                   
    return parser

class FingerCounter:
    def __init__(self):
        self.maxHands = 1
        self.detectionConfidence = 0.5
        self.trackConfidence = 0.5
        self.mpModel = mp.solutions.hands
        self.mpHands = self.mpModel.Hands(False, self.maxHands,
                            self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHand(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.mpHands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpModel.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 1, (0, 0, 0), cv2.FILLED)
        return lmList

    def drawImg(self, img,totalFingers):
        img = cv2.line(img, (10, 10), (160, 10), (0, 0, 0), 2)
        img = cv2.line(img, (10, 40), (160, 40), (0, 0, 0), 2)
        img = cv2.line(img, (10, 70), (160, 70), (0, 0, 0), 2)
        img = cv2.line(img, (10, 100), (160, 100), (0, 0, 0), 2)
        img = cv2.line(img, (10, 130), (160, 130), (0, 0, 0), 2)
        img = cv2.line(img, (10, 160), (160, 160), (0, 0, 0), 2)
        img = cv2.line(img, (10, 10), (10, 160), (0, 0, 0), 2)
        img = cv2.line(img, (50, 10), (50, 160), (0, 0, 0), 2)
        img = cv2.line(img, (120, 10), (120, 160), (0, 0, 0), 2)
        img = cv2.line(img, (160, 10), (160, 160), (0, 0, 0), 2)
        cv2.putText(img, str(1), (20, 35), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), 3)
        cv2.putText(img, str(2), (20, 65), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), 3)
        cv2.putText(img, str(3), (20, 95), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), 3)
        cv2.putText(img, str(4), (20, 125), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), 3)
        cv2.putText(img, str(5), (20, 155), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), 3)
        cv2.putText(img, str("F"), (60, 35), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 255), 2)
        cv2.putText(img, str("B"), (60, 65), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 255), 2)
        cv2.putText(img, str("L"), (60, 95), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 255), 2)
        cv2.putText(img, str("R"), (60, 125), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 255), 2)
        cv2.putText(img, str("BR"), (60, 155), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 255), 2)

        if totalFingers == 1:                    
            cv2.rectangle(img, (120, 10), (160, 40), (0, 255, 0), cv2.FILLED)
        elif totalFingers == 2:
            cv2.rectangle(img, (120, 40), (160, 70), (0, 255, 0), cv2.FILLED)
        elif totalFingers == 3:
            cv2.rectangle(img, (120, 70), (160, 100), (0, 255, 0), cv2.FILLED)
        elif totalFingers == 4:
            cv2.rectangle(img, (120, 100), (160, 130), (0, 255, 0), cv2.FILLED)
        elif totalFingers == 5:
            cv2.rectangle(img, (120, 130), (160, 160), (0, 255, 0), cv2.FILLED)
        
        return img

class Teleop:
    def __init__(self, args):
        self.height = 480
        self.width = 640
        self.cap = cv2.VideoCapture(args.input)
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)
        self.publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self.fingerCounter = FingerCounter()
        self.moveBindings = {'1':[1,0],
                             '2':[-1,0],
                             '3':[0,1],
                             '4':[0,-1],
                             '5':[0,0]}
                             
    def publishTwish(self, action):
        speed = 0.2
        turn = 0.2
        x = 0
        th = 0 
        if action == 1:
            x = self.moveBindings['1'][0]
            th = self.moveBindings['1'][1]
        elif action == 2:
            x = self.moveBindings['2'][0]
            th = self.moveBindings['2'][1]
        elif action == 3:
            x = self.moveBindings['3'][0]
            th = self.moveBindings['3'][1]
        elif action == 4:
            x = self.moveBindings['4'][0]
            th = self.moveBindings['4'][0]
        elif action == 5:
            x = self.moveBindings['5'][0]
            th = self.moveBindings['5'][1]
        target_speed = speed * x
        target_turn = turn * th

        twist = Twist()
        twist.linear.x = target_speed; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = target_turn
        self.publisher.publish(twist)

    def grapFrame(self):
        pTime = 0
        while True:
            success, img = self.cap.read()
            
            if success != True:
                break
            
            img = self.fingerCounter.findHand(img)
            lmList = self.fingerCounter.findPosition(img)
            
            tipIds = [4, 8, 12, 16, 20]

            if len(lmList) != 0:
                fingers = []
                
                # Thumb
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # 4 Fingers
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                totalFingers = fingers.count(1)
                img = self.fingerCounter.drawImg(img, totalFingers)
                self.publishTwish(totalFingers)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                        3, (255, 0, 0), 3)

            cv2.imshow("Image", img)
            cv2.waitKey(1)


def main():
    args = build_argparser().parse_args()
    rospy.init_node('teleop')
    teleop = Teleop(args)
    teleop.grapFrame()

if __name__ == "__main__":
    main()
        