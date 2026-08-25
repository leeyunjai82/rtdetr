# rtdetr — 인수인계 문서

**목표: Ultralytics YOLO와 사용성이 똑같은, Apache-2.0 RT-DETR 패키지.**
학습(PyTorch) + 추론(OpenVINO). `pip install rtdetr`.

이 문서는 ovkit 세션(claude.ai/code/session_013ZBoZjqpfewCkhmvHYAe1d)에서 넘어온
인수인계입니다. 함께 온 `rtdetr_pkg/`가 시드 코드입니다.

## 왜 만드나
- Ultralytics(YOLO)는 코드·가중치가 AGPL-3.0이라 배포 제품에 못 넣는다.
- 이 패키지는 네트워크·학습·추론 전부 자체 구현 Apache-2.0. "AGPL 없는 Ultralytics 대체"가 포지션.
- ovkit(중고생용 AI 라이브러리)이 나중에 이 패키지를 의존성으로 쓴다
  (지금 ovkit `src/ovkit/rtdetr/`에 같은 코드가 있는데, 이 저장소가 원본이 되면 ovkit 쪽은 의존성 호출로 교체 예정).

## 시드 코드 (동작 확인됨)
```
rtdetr_pkg/
  nn/          ResNet 백본(r18/r34/r50) + HybridEncoder + RTDETRDecoder — 자체 구현
  data/        YOLO 형식 데이터셋(data.yaml + images/ + labels/*.txt) 로더
  utils/       loss(헝가리안 매칭 SetCriterion), ops
  trainer.py   AdamW + warmup/cosine + AMP, runs/train/weights/best.pt
  validator.py COCO식 mAP50 / mAP50-95
  exporter.py  torch → ONNX(opset17) → OpenVINO IR (+ labels.txt, names.json)
  __init__.py  RTDETR 파사드 — ovkit 의존이 섞여 있으니 아래대로 독립시킬 것
test_rtdetr_train.py  참고용 테스트 (ovkit import 부분은 갈아끼움)
```
`__init__.py`의 `from ..core...` import와 ovkit Results 연동을 걷어내고,
원조 zip에 있던 자체 OVPredictor/Results(간단한 boxes.xyxy/conf/cls + plot/save)를 복원할 것.
(ovkit 없이 서야 함 — 의존성은 numpy/opencv/openvino/pyyaml, [train]에 torch/torchvision/scipy/onnx만.)

## Ultralytics 사용성 패리티 체크리스트
이게 "사용성 똑같이"의 정의다. 하나씩 테스트로 고정할 것.

```python
from rtdetr import RTDETR

model = RTDETR("rtdetr-r18")            # 사전학습 COCO 가중치 자동 다운로드
model = RTDETR("best.pt")               # 내 체크포인트
results = model("bus.jpg")              # list[Results]
results = model(source, conf=0.5, stream=True)   # 제너레이터
r = results[0]
r.boxes.xyxy / .conf / .cls / .xywh / .xyxyn / .xywhn
r.names; r.plot(); r.save(); r.show()
model.predict(source, save=True)        # runs/detect/predict*/ 에 저장
model.track("video.mp4")                # IoU 트래커 → boxes.id
model.train(data="data.yaml", epochs=100, imgsz=640, batch=8,
            device=0, workers=4, project="runs", name="train",
            resume=True, patience=50, lr0=1e-4, seed=0)
metrics = model.val(data="data.yaml")   # metrics.box.map50, metrics.box.map
model.export(format="openvino", half=True, imgsz=640)
```

CLI는 **ultralytics처럼 key=value** (원조 zip의 cli.py 방식 복원):
```
rtdetr predict model=rtdetr-r18 source=bus.jpg conf=0.5
rtdetr train   model=rtdetr-r18 data=data.yaml epochs=100
rtdetr val     model=best.pt data=data.yaml
rtdetr export  model=best.pt format=openvino half=true
```

source 허용 범위(ultralytics와 동일): 이미지 경로/글롭/폴더/URL/ndarray/PIL/영상/웹캠(0)/스트림.
verbose 로그 톤도 비슷하게: `image 1/1 ... 640x640 2 persons, 1 car, 12.3ms`.

## 사전학습 가중치 — 즉시 사용성의 핵심
`RTDETR("rtdetr-r18")`이 바로 동작해야 ultralytics 느낌이 난다.
- 원저자 lyuwenyu/RT-DETR가 COCO 사전학습 PyTorch 가중치를 **Apache-2.0**으로 배포함.
- 그 state_dict를 우리 `RTDETRNet` 레이어명으로 매핑 변환하는 스크립트(`tools/convert_official.py`)를 만들고,
  변환된 .pt + export IR을 HF `leeyunjai/ovkit-models`의 `rtdetr-r18/…` 경로에 업로드.
- 다운로드는 원조 zip의 stdlib 방식(urllib, 캐시 ~/.rtdetr) 재사용.

## 지켜야 할 선
- Ultralytics 코드·가중치를 한 줄도 보지 말 것(클린룸 유지). API "모양"만 같게.
- 이미 잡은 버그: export IR은 sigmoid 적용된 확률을 내므로, 추론에서 **sigmoid를 다시 걸면 안 됨**
  (점수 범위 [0,1]이면 건너뛰기 — predictor에 이미 반영돼 있음, 유지).
- exporter는 IR 옆에 labels.txt(줄당 클래스명)를 씀 — ovkit이 이걸로 클래스명을 읽는다. 유지.
- README 첫 화면: "Ultralytics-style API, 100% Apache-2.0" + 설치→추론→학습 3블록.

## 배포
- PyPI `rtdetr` (확인: 비어 있음, 2026-08-24). trusted publishing + tag `v*`.
- 버전 0.1.0부터. ovkit이 `ovkit[train] = ["rtdetr[train]"]`으로 갈아타는 건 이 저장소 안정화 후 ovkit 쪽 작업.
