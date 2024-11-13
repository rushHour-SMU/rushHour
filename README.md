
```markdown
# Gym 환경 테스트 프로젝트

이 프로젝트는 `gym` 환경을 테스트하고, `LunarLander-v2` 환경을 실행하는 예제 코드입니다.

## 1. 프로젝트 클론

먼저, GitHub에서 프로젝트를 클론합니다. 터미널을 열고 아래 명령어를 실행하세요:

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

- **Linux/macOS**:
  
  ```bash
  source venv/bin/activate
  ```

- **Windows**:

  ```bash
  .\venv\Scripts\activate
  ```

활성화가 되면, 터미널 프롬프트 앞에 `(venv)`가 표시됩니다.

## 3. 패키지 설치

`requirements.txt` 파일에 정의된 의존성 패키지들을 설치하려면 아래 명령어를 실행합니다:

```bash
pip install -r requirements.txt
```

이 명령어는 `gym`, `numpy`, `pygame` 등의 필수 패키지를 설치합니다.

## 4. `gym` 환경 실행

이제 `gym` 환경을 실행할 준비가 되었습니다. `LunarLander-v2` 환경을 테스트하는 코드 예시는 다음과 같습니다:

```python
import gym

# LunarLander-v2 환경 불러오기
env = gym.make('LunarLander-v2', render_mode='human')

N_EPISODES = 5
for episode in range(1, N_EPISODES + 1):
    reset_output = env.reset()
    state = reset_output[0]
    info = reset_output[1]

    score = 0.0
    t = 0

    while True:
        action = env.action_space.sample()  # 랜덤 액션
        state, reward, done, info = env.step(action)
        score += reward
        t += 1
        if done:
            break

    print(f'Episode {episode}: {t} steps, score={score:.1f}')

env.close()
```

이 코드를 실행하면 `LunarLander-v2` 환경에서 5번의 에피소드를 랜덤 액션으로 실행하고, 각 에피소드의 점수를 출력합니다.

## 5. 가상환경 비활성화

작업이 끝난 후, 가상환경을 비활성화하려면 다음 명령어를 실행합니다:

```bash
deactivate
```

## 6. Gitflow 브랜치 관리

이 프로젝트는 `Gitflow`를 사용하여 브랜치를 관리합니다. Gitflow를 사용하여 브랜치를 생성하고 종료하는 방법은 아래와 같습니다.

### 기능 브랜치 생성

새로운 기능을 개발할 때는 `feature` 브랜치를 생성합니다:

```bash
git flow feature start gym-test
```

### 기능 브랜치 종료

기능 작업을 완료한 후, `feature` 브랜치를 종료하고 `develop` 브랜치로 병합합니다:

```bash
git flow feature finish gym-test
```

그 후, `develop` 브랜치를 원격에 푸시합니다:

```bash
git push origin develop
```

---

이 파일을 통해 프로젝트를 처음 시작하는 사람들에게 가상환경 설정과 `gym` 환경 실행 방법을 제공하며, Gitflow 브랜치 관리도 함께 안내합니다.
```

### 설명:
- **프로젝트 클론**: GitHub에서 프로젝트를 클론하고 디렉토리로 이동하는 방법을 제공합니다.
- **가상환경 설정**: Python 가상환경을 만들고 활성화하는 방법을 안내합니다.
- **패키지 설치**: `requirements.txt`를 사용하여 필요한 패키지를 설치하는 방법을 제공합니다.
- **`gym` 환경 실행**: `LunarLander-v2` 환경을 테스트하는 코드 예시를 제공합니다.
- **가상환경 비활성화**: 작업이 끝난 후 가상환경을 비활성화하는 방법을 안내합니다.
- **Gitflow 사용**: `Gitflow`를 사용하여 기능 브랜치를 관리하는 방법을 설명합니다.

이 내용을 `README.md`에 넣으면, 프로젝트를 설정하는 방법부터 코드 실행, 브랜치 관리까지 포함된 완전한 지침서를 만들 수 있습니다.