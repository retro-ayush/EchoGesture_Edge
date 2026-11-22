"""Offline camera HUD with explicit isolated-word capture boundaries."""
import argparse
from time import perf_counter
import numpy as np
from echogesture.extract import Extractor
from echogesture.inference import Predictor


def main():
    import cv2
    p=argparse.ArgumentParser();p.add_argument("--camera",type=int,default=0)
    p.add_argument("--model",default="models/echogesture.int8.onnx")
    p.add_argument("--models",default="models")
    p.add_argument("--threshold",type=float,default=None,help="Default: calibrated value from model JSON (echogesture.validate), else 0.6")
    a=p.parse_args(); predictor=Predictor(a.model); extractor=Extractor(a.models)
    if a.threshold is None:a.threshold=float(predictor.meta.get("confidence_threshold",.6))
    print(f"Confidence threshold {a.threshold:.2f}")
    camera=cv2.VideoCapture(a.camera)
    active=False; sequence=[]; caption="Ready to capture a sign"; detail=""; start=perf_counter()
    try:
        if not camera.isOpened():raise RuntimeError("Camera unavailable")
        while True:
            ok,frame=camera.read()
            if not ok:break
            if active:
                sequence.append(extractor.frame(frame,(perf_counter()-start)*1000))
                if len(sequence)>300:
                    active=False; sequence=[]; caption="Capture too long. Try one word."
            # Mirror the display only. Models always receive original frames.
            display=cv2.flip(frame,1); h,w=display.shape[:2]
            panel=np.zeros((195,max(w,760),3),np.uint8);panel[:]=(30,24,18)
            cv2.putText(panel,"ECHOGESTURE EDGE",(24,34),cv2.FONT_HERSHEY_SIMPLEX,.7,(180,220,220),2)
            cv2.putText(panel,"CAPTURING" if active else caption,(24,83),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            cv2.putText(panel,detail,(24,119),cv2.FONT_HERSHEY_SIMPLEX,.47,(190,190,190),1)
            cv2.putText(panel,"SPACE: start / finish one sign    C: clear    Q: quit",(24,157),cv2.FONT_HERSHEY_SIMPLEX,.48,(190,220,190),1)
            cv2.putText(panel,"Local processing | ASL word prototype | No video recording",(24,182),cv2.FONT_HERSHEY_SIMPLEX,.43,(160,160,160),1)
            view=np.zeros((h,panel.shape[1],3),np.uint8);view[:,(panel.shape[1]-w)//2:(panel.shape[1]+w)//2]=display
            cv2.imshow("EchoGesture Edge",np.vstack([view,panel]))
            key=cv2.waitKey(1)&255
            if key==ord("q"):break
            if key==ord("c"):caption="Ready to capture a sign";detail="";sequence=[];active=False
            if key==32:
                if not active:sequence=[];active=True;detail="Sign one vocabulary word, then press SPACE."
                else:
                    active=False
                    if len(sequence)<8:caption="Capture more frames";continue
                    result=predictor.classify(np.stack(sequence))
                    confidence=result["confidence"]
                    caption=result["prediction"] if confidence>=a.threshold else "Uncertain - please repeat"
                    detail=f"Score {confidence:.2f} | Classifier {result['classifier_latency_ms']:.1f} ms | {len(sequence)} captured frames"
                    sequence=[]
    finally:
        camera.release();extractor.close();cv2.destroyAllWindows()


if __name__ == "__main__":main()
