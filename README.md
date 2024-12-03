## 1. 프로젝트 클론

먼저, GitHub에서 프로젝트를 클론합니다:

```bash
git clone https://github.com/rushHour-SMU/rushHour.git
cd rushHour
```

## 2. 가상환경 설정

### 가상환경 만들기

프로젝트 디렉토리에서 가상환경을 생성합니다:

```bash
python3 -m venv venv
```

### 가상환경 활성화

가상환경을 활성화하려면, 아래 명령어를 사용합니다.

```bash
./activate_deactivate
```

활성화가 되면, 터미널 프롬프트 앞에 `(venv)`가 표시됩니다.

## 3. 패키지 설치

`requirements.txt` 파일에 정의된 의존성 패키지들을 설치하려면 아래 명령어를 실행합니다:

```bash
pip install -r requirements.txt
```

## 전반적으로 ipynb 파일에서 영상 시각화를 위해 모듈화해서 불러왔기에 오류가 존재 할 수 있는데 그럴때는 ipynb에서 학습 자체 결과는 확인 할 수 있다.
