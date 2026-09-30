import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ============================================================
# 1. 모델 및 토크나이저 로드
# ============================================================

MODEL_NAME = "gpt2"

# CUDA GPU를 사용할 수 있으면 GPU, 아니면 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("1. 모델 및 토크나이저 로드")
print("=" * 70)
print(f"사용 장치: {device}")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# 모델을 CPU 또는 GPU로 이동
model = model.to(device)

# 평가 모드로 설정
# Dropout 등이 비활성화되어 추론 결과가 안정적으로 동작함
model.eval()

# GPT-2에는 기본 pad_token이 없으므로 eos_token을 사용
tokenizer.pad_token = tokenizer.eos_token

print("GPT-2 모델 로드 완료\n")


# ============================================================
# 2. 입력 문장 전처리
# ============================================================

print("=" * 70)
print("2. 입력 문장 전처리")
print("=" * 70)

# GPT-2는 주로 영어 데이터로 학습되었으므로 영어 문장을 사용
prompt = "Artificial intelligence will change the future because"

print(f"입력 문장: {prompt}")

# 문자열을 GPT-2가 처리할 수 있는 Token ID로 변환
inputs = tokenizer.encode(
    prompt,
    return_tensors="pt"
).to(device)

print(f"Token ID: {inputs}")
print(f"Token 개수: {inputs.shape[1]}\n")


# ============================================================
# 텍스트 생성 함수
# ============================================================

def generate_text(
    input_ids,
    do_sample=False,
    temperature=1.0,
    top_p=1.0,
    max_length=60
):
    """
    GPT-2를 이용하여 텍스트를 생성한다.

    do_sample=False:
        확률 분포에서 가장 가능성이 높은 토큰을 선택한다.
        같은 입력이면 일반적으로 같은 결과가 나온다.

    do_sample=True:
        확률 분포를 이용하여 토큰을 샘플링한다.
        실행할 때마다 결과가 달라질 수 있다.

    temperature:
        확률 분포의 날카로움을 조절한다.
        낮을수록 보수적이고 높은 확률의 토큰을 선호한다.
        높을수록 다양한 토큰이 선택될 가능성이 커진다.

    top_p:
        누적 확률이 top_p가 될 때까지의 후보 토큰만 사용한다.
        예를 들어 0.9라면 확률 질량의 상위 90% 후보에서 샘플링한다.
    """

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id
        )

    # 생성된 Token ID를 사람이 읽을 수 있는 문자열로 변환
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ============================================================
# 3. 기본 생성 결과
# ============================================================

print("=" * 70)
print("3. 기본 생성 결과")
print("=" * 70)

# Sampling을 사용하지 않는 Greedy Decoding
basic_result = generate_text(
    inputs,
    do_sample=False,
    max_length=60
)

print(basic_result)
print()


# ============================================================
# 4. 샘플링 설정 비교
# ============================================================

print("=" * 70)
print("4. 샘플링 설정 비교")
print("=" * 70)

# ------------------------------------------------------------
# 4-1. do_sample=False
# ------------------------------------------------------------

print("\n[do_sample=False]")
print("가장 확률이 높은 토큰을 계속 선택합니다.\n")

for i in range(2):
    result = generate_text(
        inputs,
        do_sample=False,
        max_length=60
    )

    print(f"실행 {i + 1}:")
    print(result)
    print()


# ------------------------------------------------------------
# 4-2. do_sample=True
# ------------------------------------------------------------

print("\n[do_sample=True]")
print("확률 분포에 따라 토큰을 샘플링합니다.\n")

for i in range(2):
    result = generate_text(
        inputs,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        max_length=60
    )

    print(f"실행 {i + 1}:")
    print(result)
    print()


# ============================================================
# 5. Seed 고정 실험
# ============================================================

print("=" * 70)
print("5. Seed 고정 실험")
print("=" * 70)

SEED = 42

# ------------------------------------------------------------
# Seed를 매번 동일하게 설정
# ------------------------------------------------------------

print("\n[Seed를 매번 42로 고정]")

for i in range(2):

    # 난수 생성기의 상태를 동일하게 초기화
    torch.manual_seed(SEED)

    # CUDA 사용 시 CUDA 난수도 고정
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    result = generate_text(
        inputs,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        max_length=60
    )

    print(f"\n실행 {i + 1}:")
    print(result)


# ------------------------------------------------------------
# Seed를 고정하지 않은 경우
# ------------------------------------------------------------

print("\n\n[Seed를 고정하지 않은 경우]")

# 난수 상태를 새로운 값으로 변경
torch.seed()

for i in range(2):

    result = generate_text(
        inputs,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        max_length=60
    )

    print(f"\n실행 {i + 1}:")
    print(result)


# ============================================================
# 실험 종료
# ============================================================

print("\n" + "=" * 70)
print("실험 완료")
print("=" * 70)