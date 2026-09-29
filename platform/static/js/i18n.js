// easydetect platform — 한국어 / English.
//
// Korean is the source text; English comes from EN below. Text in the page
// (markup, or HTML that script writes) is swapped whole: a text node or a
// title/placeholder whose trimmed text is a key. Strings the script builds
// with values in them go through T`…${x}…`, whose holes are {0}, {1}, … in
// the key. Progress lines the server writes in Korean fall back to FRAGMENTS.
// Anything without an entry stays Korean rather than breaking.
(() => {
  const EN = {
    // header, steps, panels
    "데이터셋": "Datasets", "학습": "Train", "추론": "Predict", "모델": "Models",
    "한국어 / English": "한국어 / English",
    "PyTorch가 이 GPU를 못 써요. 학습이 CPU로 돌아요. CUDA 빌드 torch를 깔아 주세요 (RTX 50 계열은 cu128 이상).":
      "PyTorch can't use this GPU, so training runs on the CPU. Install a CUDA build of torch (cu128 or newer for the RTX 50 series).",
    "데이터 올리기": "Add data", "상자 그리기": "Draw boxes", "모델 쓰기": "Use the model",
    "쉬는 중": "Idle", "CPU 전용": "CPU only", "#{0} 실행 중": "#{0} running", "대기 {0}": "{0} queued",
    "사용률": "Load", "메모리": "Memory",

    // datasets
    "사진·영상·zip을 올리고, 상자를 그린 뒤 학습에 써요.": "Add photos, video or a zip, draw boxes, then train on it.",
    "이미지": "Images", "영상": "Video", "폴더": "Folder",
    "이미지를 끌어다 놓거나 눌러요": "Drop images here, or click", "여러 장 한 번에": "As many as you like",
    "영상을 끌어다 놓거나 눌러요": "Drop a video here, or click", "일정 간격으로 프레임을 뽑아요": "Frames are taken at an interval",
    "프레임 간격": "Every n frames", "최대 장수": "Max frames",
    "zip을 끌어다 놓거나 눌러요": "Drop a zip here, or click",
    "Roboflow 등 YOLO 형식, 또는 images/ + labels/": "YOLO format as Roboflow exports it, or images/ + labels/",
    "이 서버의 폴더 경로": "Folder path on this server", "클래스": "Classes",
    "쉼표로 나눠요. 비우면 data.yaml을 읽거나 알아서 정해요.": "Comma-separated. Leave empty to read data.yaml or work them out.",
    "이름": "Name", "자동": "Automatic", "넣을 곳": "Add to", "새 데이터셋": "New dataset",
    "데이터 추가": "Add data", "내 데이터셋": "My datasets",
    "카드 메뉴에서 '병합 선택'으로 두 개 이상 골라요": "Tick 'Select to merge' in two or more card menus",
    "선택 병합": "Merge selected", "선택 병합 ({0})": "Merge selected ({0})",
    "{0}개 파일 · {1} 외": "{0} files · {1} and more",
    "이미지 {0}": "{0} images", "라벨 {0}/{1}": "labelled {0}/{1}", "라벨 {0}": "{0} labelled",
    "{0}장 추가": "{0} added", "클래스 {0}": "classes {0}",
    "더 보기": "More", "통계 보기": "Statistics", "전체 자동 라벨": "Auto-label all",
    "zip으로 내보내기": "Export as zip", "복제": "Duplicate", "병합 선택": "Select to merge", "삭제": "Delete",
    "라벨": "Labelled", "이 데이터로 학습": "Train on this",
    "상자를 그려야 학습할 수 있어요": "Draw some boxes first",
    "아직 데이터셋이 없어요.": "No datasets yet.", "위에서 사진이나 zip을 올려 봐요.": "Add photos or a zip above to start.",
    "상자 합계": "Boxes in total",
    "라벨 없음 {0}장": "{0} unlabelled", "빈 라벨(배경) {0}장": "{0} empty (background)",
    "아주 작은 상자 {0}개": "{0} tiny boxes", "클래스 범위 밖 {0}개": "{0} boxes with an unknown class",
    "병합된 데이터셋 이름": "Name for the merged dataset",
    "{0}장을 합쳤어요 · 클래스 {1}": "Merged {0} images · classes {1}",
    "데이터셋 #{0}과 그 작업 기록을 지울까요?": "Delete dataset #{0} and its jobs?",
    "이미지를 골라 주세요": "Pick some images", "영상을 골라 주세요": "Pick a video", "zip을 골라 주세요": "Pick a zip",
    "폴더 경로를 적어 주세요": "Enter a folder path", "올리는 중…": "Uploading…", "복제했어요": "Duplicated",

    // training
    "COCO 가중치에서 시작해 내 데이터에 맞춰요. 끝나면 OpenVINO IR도 만들어 둬요.":
      "Starts from the COCO weights and fits them to your data. An OpenVINO IR is exported at the end.",
    "상자를 그린 데이터셋이 아직 없어요.": "No dataset has boxes yet.", "데이터셋 추가하기": "Add a dataset",
    "목표": "Goal", "빠르게": "Fast", "균형": "Balanced", "정확하게": "Accurate",
    "시작 모델": "Start from", "n 가장 가벼움 · s 기본 · m · l · x 가장 정확": "n lightest · s default · m · l · x most accurate",
    "에폭": "Epochs", "배치": "Batch", "고급 설정": "Advanced", "입력 크기": "Input size", "검증 비율": "Validation share",
    "백본 고정": "Freeze backbone", "예 (빨라요)": "Yes (faster)", "아니오": "No", "장치": "Device", "학습 시작": "Start training",
    "{0}에폭": "{0} epochs", "배치 {0}": "batch {0}",
    "안 나아지면 멈추기": "Stop when it stops improving", "에폭 동안 mAP가 그대로면 멈춰요": "epochs with no better mAP, then stop",
    "가장 좋았던 가중치가 남아요. 0이면 끝까지 해요.": "The best weights are kept. 0 runs every epoch.",
    "{0}에폭 정체 시 멈춤": "stop after {0} flat epochs",
    "데이터셋을 골라 주세요": "Pick a dataset",
    "이 데이터셋은 #{0}이(가) 벌써 대기 중이에요. 하나 더 넣을까요?": "#{0} is already queued for this dataset.\nQueue another?",
    "이 데이터셋은 #{0}이(가) 벌써 학습 중이에요. 하나 더 넣을까요?": "#{0} is already training on this dataset.\nQueue another?",
    "#{0}을(를) 대기열에 넣었어요": "#{0} queued",

    // predict
    "모델을 데이터셋·폴더·영상·웹캠에 돌려 봐요.": "Run a model over a dataset, a folder, a video or a webcam.",
    "확신 정도 (conf)": "Confidence (conf)", "낮추면 더 많이 찾고, 높이면 확실한 것만 남아요.": "Lower finds more; higher keeps only the sure ones.",
    "대상": "Source", "웹캠": "Webcam", "/data/새-촬영분": "/data/new-shots",
    "상자를 그린 mp4로 돌려받아요": "You get an mp4 back with the boxes drawn",
    "카메라 시작": "Start camera", "정지": "Stop", "추론 실행": "Run",
    "데이터셋이 없어요": "There is no dataset", "보내는 중…": "Sending…", "#{0}을(를) 시작했어요": "#{0} started",
    "카메라는 https에서 열려요": "The camera opens over https",
    "처음 뜨는 인증서 경고에서 '고급 → 계속'을 눌러요": "On the certificate warning, choose 'Advanced → Continue' once",
    "카메라는 https나 localhost에서만 열려요. 서버를 --https-port 와 함께 켜 주세요.":
      "Browsers only open a camera on https or localhost. Start the server with --https-port.",
    "카메라가 안 열려요: {0}": "The camera didn't open: {0}", "찾은 물체가 없어요": "Nothing found",
    "내 학습 결과": "My runs", "등록된 모델": "Registered models", "COCO 사전학습 · 80 클래스": "COCO pretrained · 80 classes",

    // models
    "학습 결과에 이름을 붙여 두면 자동 라벨과 추론에서 골라 써요.": "Name a run to pick it for auto-labelling and prediction.",
    "외부 모델 올리기": "Add a model from elsewhere", "IR은 두 파일을 함께 골라요": "Pick both files of an IR",
    "이름 (비우면 파일 이름)": "Name (default: the file name)", "올리기": "Upload",
    "코드에서 쓰기": "Use it in code", "어느 PC든": "On any PC,",
    "하나면 돼요. 추론에는 PyTorch가 필요 없어요. 학습한 모델의 경로가 채워진 코드는 작업을 열면 나와요.":
      "is all it takes; prediction needs no PyTorch. Open a job for code with your model's path filled in.",
    "복사": "Copy", "복사했어요": "Copied", "코드": "Code",
    "등록한 모델이 아직 없어요.": "No models registered yet.", "작업을 열고 \"모델로 등록\"을 눌러요.": "Open a job and press \"Register model\".",
    "이 모델을 목록에서 뺄까요? 파일은 그대로 남아요.": "Remove this model from the list? Its files stay.",
    "파일을 골라 주세요": "Pick a file", "올렸어요": "Uploaded", "모델 이름": "Model name",

    // the job panel
    "시작하기": "Getting started", "닫기": "Close",
    "네 단계면 내 물건을 찾는 모델이 생겨요. 작업을 누르면 여기서 결과를 봐요.":
      "Four steps to a model that finds your objects. Click a job to see it here.",
    "사진·영상·Roboflow zip": "Photos, video, a Roboflow zip", "자동 라벨로 초안을 받아요": "Auto-label drafts them for you",
    "COCO에서 시작해 몇 분": "Minutes, starting from COCO", "추론 · 배포": "Predict · deploy", "웹캠·영상 · OpenVINO IR": "Webcam, video · OpenVINO IR",
    "중지": "Stop", "이어서 학습": "Train more", "평가": "Evaluate", "모델로 등록": "Register model", "다운로드": "Download",
    "로그": "Log", "결과 이미지 zip": "Result images (zip)", "결과 영상": "Result video",
    "이 작업과 결과 파일을 지워요": "Deletes this job and its files",
    "학습 곡선": "Training curve", "에폭별 기록": "Per epoch", "에폭별 기록 ({0})": "Per epoch ({0})", "초": "s",
    "이 모델로 시험해 보기": "Try this model", "이미지 한 장": "One image", "끌어다 놓거나 눌러요": "Drop it here, or click",
    "평가 결과": "Evaluation", "학습 정보": "Run details", "명령줄": "Command line", "다른 PC": "Another PC",
    "저장소": "Repository", "저장소 안 폴더": "Folder in the repository",
    "best.pt도 넣기 — 받은 사람이 이어서 학습할 수 있어요 (용량이 늘어요)": "Include best.pt — so others can keep training it (larger)",
    "업로드용 zip 받기": "Download the upload zip", "README 보기": "Show README", "업로드 명령 보기": "Show upload commands",
    "작업": "Jobs", "전체": "All", "자동 라벨": "Auto-label", "{0}개": "{0}",
    "학습·자동 라벨·평가·추론이": "Training, auto-labelling, evaluation", "여기 차례로 쌓여요.": "and predictions line up here.",
    "이 종류의 작업은 아직 없어요.": "No jobs of this kind yet.",
    "대기": "Queued", "실행 중": "Running", "내보내는 중": "Exporting", "완료": "Done", "실패": "Failed", "중지됨": "Stopped",
    "{0} 걸림": "took {0}", "곧 시작해요": "Starting soon", "기다리는 중이에요. #{0}이(가) 끝나면 시작해요": "Waiting for #{0} to finish",
    "몇 에폭 더 학습할까요?": "How many more epochs?",
    "\"{0}\"(으)로 등록했어요. 모델 탭과 추론에서 골라요": "Registered as \"{0}\". Pick it under Models and Predict.",
    "#{0} 작업과 결과 파일(가중치, IR, 로그)을 지울까요?": "Delete job #{0} and its files (weights, IR, log)?",
    "이어서 학습·평가 {0}개도 같이 지워져요.": "Its {0} continuations and evaluations go with it.",
    "시험할 이미지를 골라 주세요": "Pick an image to try", "추론 중…": "Predicting…",
    // run details
    "시작 가중치": "Started from", "학습 데이터": "Training data", "검증 데이터": "Validation data",
    "학습하는 파라미터": "Trained parameters", "옵티마이저": "Optimizer", "스케줄": "Schedule", "증강": "Augmentation",
    "조기 종료": "Early stopping", "최고 에폭": "Best epoch", "마지막 loss": "Last loss", "버전": "Versions",
    "COCO 사전학습 가중치": "COCO pretrained weights", "ImageNet 백본만 (COCO를 못 받음)": "ImageNet backbone only (COCO unavailable)",
    "처음부터 (무작위)": "From scratch (random)", "직접 준 네트워크": "A network you passed in", "이전 학습 · {0}": "Earlier run · {0}",
    "{0}장 · 상자 {1}": "{0} images · {1} boxes", "배경 {0}": "{0} background", "파라미터 {0}": "{0} parameters",
    "{0} 고정": "{0} frozen", "백본 {0}": "backbone {0}", "일찍 멈춤": "stopped early", "에폭당 {0}": "{0} per epoch",
    "{0}에폭부터": "from epoch {0}", "모델 구조": "Architecture", "이어서 학습한 곳": "Resumed", "학습 상자": "Train boxes", "검증 상자": "Val boxes",
    // results
    "상자": "Boxes", "상자 {0}": "{0} boxes", "지금까지 그린 결과 {0}장": "{0} results drawn so far",
    "처음 {0}장만 보여요. 전부는 다운로드 → 결과 이미지 zip": "Showing the first {0}. All of them: Download → Result images (zip)",
    "초록 상자가 정답, 색 상자가 예측이에요. 빨간 숫자는 놓친 이미지예요.": "Green boxes are the truth, coloured ones the prediction. Red counts mark missed images.",
    "정답 {0}": "truth {0}", "찾음 {0}": "found {0}",
    "{0}시간 {1}분": "{0} h {1} min", "{0}분 {1}초": "{0} min {1} s", "{0}초": "{0} s",
    // code comments
    "상자를 그린 이미지": "the picture with its boxes", "영상 · 웹캠(0) · RTSP: 한 프레임씩 받기": "video · webcam (0) · RTSP: one frame at a time",
    "이미지 · 폴더 · 영상 — 결과는 runs/detect/predict/": "image · folder · video — results in runs/detect/predict/",
    "웹캠 창으로 (q 또는 Esc로 종료)": "in a webcam window (q or Esc quits)",
    "다운로드 → OpenVINO IR 로 zip을 받아 풀어요 ({0} · .bin · labels.txt)": "Download → OpenVINO IR, and unzip it ({0} · .bin · labels.txt)",
    "그 PC에는 추론용만 설치해요 — PyTorch 없이 돼요": "on that PC install the inference package only — no PyTorch needed",
    "파이썬에서": "in Python", "Intel NPU면 \"NPU\"": "\"NPU\" on an Intel NPU",
    "\"업로드용 zip 받기\" → 풀면 {0}/ 폴더 (모델 · labels.txt · README.md)": "\"Download the upload zip\" → unzips to {0}/ (model · labels.txt · README.md)",
    "한 번만: 쓰기 권한 토큰으로 로그인": "once: log in with a token that can write",
    "폴더째 올리기 — 저장소의 {0}/ 로 들어가요": "upload the folder — it lands in {0}/ in the repo",
    "받는 쪽 (README에도 같은 코드가 들어 있어요)": "on the receiving end (the README has this code too)",
    "처음 한 번 가중치를 받아요 (COCO 80 클래스)": "downloads the weights once (COCO, 80 classes)",
    "학습한 모델은 그 경로를 그대로 넣어요": "a model you trained: pass its path",

    // label page
    "상자 그리기 — easydetect": "Draw boxes — easydetect", "{0} — 상자 그리기": "{0} — draw boxes",
    "플랫폼으로": "Back to the platform", "클래스 추가": "Add a class", "추가": "Add", "폴더별로 보기": "Show one folder",
    "상자 없는 것만 보기": "Only images without boxes", "이전 (←)": "Previous (←)", "이전": "Previous",
    "다음 (→)": "Next (→)", "다음": "Next", "모델이 상자 초안을 그려요": "A model drafts the boxes",
    "이전 이미지의 상자를 그대로 가져와요 (C)": "Copy the previous image's boxes (C)", "이전 상자 복사": "Copy previous boxes",
    "되돌리기": "Undo", "모두 지우기": "Clear all", "이 이미지와 상자를 데이터셋에서 지워요": "Removes this image and its boxes from the dataset",
    "이미지 삭제": "Delete image",
    "끌어서 그려요 · 상자 안을 끌면 옮겨요 · 모서리를 끌면 크기가 바뀌어요 ·": "Drag to draw · drag inside a box to move it · drag a corner to resize ·",
    "클래스 ·": "class ·", "지우기 ·": "delete ·", "되돌리기 ·": "undo ·", "넘기기 ·": "previous / next ·",
    "이전 상자 복사 · 휠 확대 ·": "copy previous boxes · wheel zooms ·", "+끌기로 화면 이동 · 저장은 저절로 돼요": "+drag pans · saving is automatic",
    "모든 폴더 ({0})": "All folders ({0})", "(최상위)": "(top level)", "상자 {0}개": "{0} boxes",
    "빈 라벨 (배경 이미지)": "Empty label (background image)", "아직 상자가 없어요": "No boxes yet",
    "다 그린 것 {0}/{1}": "done {0}/{1}", "복사할 이전 상자가 없어요": "No previous boxes to copy",
    "자동 라벨 중…": "Auto-labelling…", "자동 라벨이 안 됐어요: {0}": "Auto-label failed: {0}",
    "{0} 을(를) 데이터셋에서 지울까요?": "Remove {0} from the dataset?",
  };

  // what the server writes while a job runs: "에폭 3/50 · 배치 12/40 · 남은 시간 약 4분"
  const FRAGMENTS = [
    [/(\d+)시간/g, "$1 h"], [/(\d+)분/g, "$1 min"], [/(\d+)초/g, "$1 s"],
    [/남은 시간 계산 중/g, "estimating time left"], [/남은 시간 약 ([^·]+?)\s*$/, "~$1 left"],
    [/에폭 (\d+)\/(\d+)/g, "epoch $1/$2"], [/배치 (\d+)\/(\d+)/g, "batch $1/$2"], [/검증 중/g, "validating"],
    [/프레임 (\d+)\/(\d+)/g, "frame $1/$2"], [/이미지 (\d+)\/(\d+)/g, "image $1/$2"],
    [/상자 (\d+)개/g, "$1 boxes"], [/(\d+)장 미리보기/g, "$1 previews"], [/(\d+)장/g, "$1 images"],
    [/모델 불러오는 중…/g, "loading the model…"], [/#(\d+) 이어서 \+(\d+)에폭/g, "#$1 continued, +$2 epochs"],
    [/이미 학습은 끝났고 내보내는 중입니다/g, "training is over; exporting"], [/^업로드$/, "uploaded"], [/^서버가 다시 켜지면서 끊겼어요$/, "interrupted by a restart"],
    [/^#(\d+)이\(가\) 아직 돌고 있어요\. 먼저 멈춰 주세요$/, "#$1 is still running — stop it first"],
    [/^등록한 모델 '(.+)'이\(가\) 이 결과를 써요\. 모델 탭에서 먼저 빼 주세요$/,
     "registered model '$1' uses this run — remove it under Models first"],
  ];

  const HANGUL = /[가-힣]/;
  const ATTRS = ["title", "placeholder", "label", "aria-label"];
  const norm = (s) => s.trim().replace(/\s+/g, " ");
  const store = { get: () => { try { return localStorage.getItem("lang"); } catch { return null; } },
                  set: (v) => { try { localStorage.setItem("lang", v); } catch {} } };
  let lang = store.get() || ((navigator.language || "").toLowerCase().startsWith("ko") ? "ko" : "en");

  function english(text) {
    const key = norm(text);
    if (!HANGUL.test(key)) return null;
    let en = EN[key];
    if (en == null) {
      en = key;
      for (const [pattern, to] of FRAGMENTS) en = en.replace(pattern, to);
      if (en === key) return null;
    }
    // keep the spacing around the words: "이미지 " before a <b> stays "Images "
    return text.match(/^\s*/)[0] + en + text.match(/\s*$/)[0];
  }

  // T`이미지 ${n}` or T("쉬는 중")
  function T(strings, ...values) {
    if (typeof strings === "string") strings = [strings];
    const korean = () => strings.reduce((out, s, i) => out + (i ? values[i - 1] : "") + s, "");
    if (lang !== "en") return korean();
    const en = EN[norm(strings.reduce((k, s, i) => k + (i ? `{${i - 1}}` : "") + s, ""))];
    return en == null ? korean() : en.replace(/\{(\d+)\}/g, (_, i) => values[i]);
  }

  const ORIGINAL = new WeakMap();   // text node → its Korean
  function swap(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => n.nodeType === 1 && (n.closest("pre, script, style, [data-no-i18n]"))
        ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT });
    for (let node = root.nodeType === 3 ? root : walker.currentNode; node; node = walker.nextNode()) {
      if (node.nodeType === 3) {
        if (!ORIGINAL.has(node)) {
          if (!HANGUL.test(node.nodeValue)) continue;
          ORIGINAL.set(node, node.nodeValue);
        }
        // data-en on the parent settles a word that means two things ("모델": Models / Model)
        const ko = ORIGINAL.get(node), over = node.parentElement?.dataset.en;
        const want = lang === "en" ? over ?? english(ko) ?? ko : ko;
        if (node.nodeValue !== want) node.nodeValue = want;
      } else if (node.nodeType === 1) {
        if (node.closest("pre, script, style, [data-no-i18n]")) continue;
        for (const a of ATTRS) {
          if (!node.hasAttribute(a)) continue;
          const keep = `data-ko-${a}`;
          if (!node.hasAttribute(keep)) {
            if (!HANGUL.test(node.getAttribute(a))) continue;
            node.setAttribute(keep, node.getAttribute(a));
          }
          const ko = node.getAttribute(keep);
          node.setAttribute(a, lang === "en" ? english(ko) ?? ko : ko);
        }
      }
      if (root.nodeType === 3) break;
    }
  }

  function button() {
    const b = document.getElementById("langToggle");
    if (b) b.textContent = lang === "en" ? "한" : "EN";
  }

  function set(next) {
    lang = next; store.set(next);
    document.documentElement.lang = next;
    swap(document.body); button();
    dispatchEvent(new Event("langchange"));   // the page rewrites what its script wrote
  }

  // whatever the page adds later, in English while English is on
  new MutationObserver((records) => {
    if (lang !== "en") return;
    for (const r of records) r.addedNodes.forEach((n) => {
      if (n.nodeType === 3 ? n.parentElement && !n.parentElement.closest("pre, script, style, [data-no-i18n]") : n.nodeType === 1) swap(n);
    });
  }).observe(document.body, { childList: true, subtree: true });

  document.documentElement.lang = lang;
  if (lang === "en") swap(document.body);
  button();
  document.getElementById("langToggle")?.addEventListener("click", () => set(lang === "en" ? "ko" : "en"));

  window.T = T;
  window.i18n = { get lang() { return lang; }, set, english };
})();
