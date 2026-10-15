# 실사(photoreal) 이미지 백엔드

NovelAI(애니메이션, Danbooru 태그) 옆에 **로컬 오픈웨이트 실사 모델**을 두 번째 렌더러로 붙이고, 컷마다
어느 쪽으로 그릴지 정한다. 실사 모델은 로컬 ComfyUI 가 돌리고, DynamicChat API 서버가 릴레이한다.

```
cue (annotation pass) → createImagePromptPlan (NAI 태그 합성: 정체성·의상·연속성·프레임)
   → classifyImageScene (합성된 태그로 장면 분류) → decideImageBackend (라우팅 + 안전 하한)
   ├─ novelai  → 기존 경로 (+ Precise Reference, 선택)
   └─ photoreal → buildPhotorealPrompt (자연어) → buildComfyWorkflow → POST /photoreal/comfyui/generate
                  → 릴레이: 참조 업로드 → /prompt → /history 폴링 → /view → base64 → 자산
```

## 왜 섞는가

NAI 는 역동적인 동작과 다양한 상황을 잘 그리고, 실사 모델은 정적인 사진(인물 단독, 풍경, 사물)에 강하다.
그래서 **장면 범주별로** 라우팅한다. 기본 혼합 선택은 `portrait`, `full_body`, `scenery`, `object` 다.

## 라우팅

| 모드 | 동작 |
|---|---|
| `novelai` (NAI만) | 기존 동작. 기존 시뮬레이션의 기본값 |
| `mixed` (장면별 혼합) | `photorealCategories` 에 체크된 범주만 실사 (다중 선택) |
| `photoreal` (전부 실사) | 안전 하한에 걸리는 컷을 빼고 전부 실사 |

범주는 `imageSceneClassifier.ts` 가 **합성된 프롬프트**에서 결정한다. 작성된 태그만 보면 안 된다 —
연속성 주입이 진행 중인 체위를 얼굴 클로즈업에 다시 넣기 때문에, 작성자가 "얼굴만" 쓴 컷이 실제로는
노골적인 컷이 된다(eval 에 그 경우가 있다). 우선순위: explicit > action > fantasy > romance > (사람 없음:
object|scenery) > group > portrait|full_body.

사용 불가 백엔드는 대체된다(`fallback: true`): 실사가 꺼져 있으면 NAI 로, NAI 가 꺼져 있으면 실사로.
결정은 `job.routing` 에 1급 필드로 남는다 — 저장된 진행은 `providerPayload` 를 비우기 때문이다.

## 안전 하한 (코드로 강제, eval 로 고정)

1. `explicit` 컷은 어떤 모드에서도, 강제 재생성으로도 실사가 되지 않는다(NAI 고정, `locked`).
2. 미성년 코드 태그(loli, child, teen, 나이 표기 <18 …)가 있으면 실사 금지.
3. 학교 코드 + 선정적 태그 조합은 실사 금지.
4. 실사 프롬프트의 기본 피사체 명사는 성인("an adult woman").
5. 실사 얼굴 레퍼런스는 **DynamicChat 이 생성하고 사용자가 고른 이미지**뿐이다. 업로드 경로는 일부러 없다 —
   실존 인물 사진이 실사 얼굴 레퍼런스가 되는 것을 막는다.

NAI 동작은 바꾸지 않았다(기존 eval 이 NAI 에 앱 수준 차단이 없음을 단언한다). 하한은 실사에만 걸린다.

**하한은 샘플러에 닿는 모든 글을 읽는다 — 보내기 직전의 그래프까지.** 적대적 레드팀을 세 번 돌렸고, 매번 나온 구멍은
같은 모양이었다: 분류기가 보지 않는 글이 사진으로 간다. 그래서 읽는 곳이 계속 늘었다.

| 어디서 | 무엇을 |
|---|---|
| 계획 | 합성된 NAI 캡션, 캐릭터 `photorealPrompt`(실사 외형), 실사 스타일 글, 사용자 규정, 연속성 주입 |
| 계획 끝 | **완성된 사진 프롬프트** 자체, 그리고 **실제로 조립한 ComfyUI 그래프**의 그림 쪽 글 전부(하나씩, 이어서) |
| 설정 | 사용자 워크플로의 글자 그대로의 문구(노드 단위로, 노드를 가로질러 이어서) |
| 실행 직전 | 보낼 그래프 객체 그대로(`inspectWorkflow`) + 참조 이미지 기록 — 계획 뒤 바뀐 설정·템플릿·어휘도 반영 |
| 제작 화면 | 테스트 프롬프트(걸리면 **거절** — NovelAI 로 몰래 돌리면 Anlas 를 쓴다) |

- **조립 후를 읽는 이유**: 스타일 끝 `na` + 템플릿 `%prompt%ked` 처럼 조각마다 깨끗해도 합치면 단어가 되고,
  `%steps%` 같은 숫자 자리표시자를 문장에 넣으면("a %steps% year old girl") 나이가 된다. 그래서 숫자 자리표시자는
  숫자 입력의 값 전체로만 쓸 수 있고, `%prompt%` 는 글자와 붙일 수 없다.
- **CFG 는 1 이상**(`PHOTOREAL_MIN_CFG`). 1 미만이면 샘플러가 네거티브 쪽으로 그려 **네거티브가 그림이 된다**("naked child"
  를 네거티브에 넣고 cfg 0.1). 설정·저장값·템플릿 리터럴·링크·릴레이(`EXTRA_NODES=*` 포함) 모두에서 막는다. Turbo
  프리셋은 증류 모델이라 cfg 1 고정이다.
- **네거티브는 허용 목록이다.** 화질·결함·매체 단어(blurry, lowres, extra fingers, anime …)와 몇몇 소품(glasses, hat,
  tattoo …)만 남고 나머지는 조용히 빠진다. 차단 목록은 75개 중 70개가 새어 나갔다 — cfg > 1 에서 네거티브는 반대쪽으로
  끌기 때문에 옷(어느 언어·철자든)이나 성인·나이·체형 단어를 네거티브에 넣으면 벗은·어린 인물 쪽으로 간다.
- **사용자 워크플로**: LoadImage 는 `%imageN%`/`%ref:%` 자리표시자만(클라이언트 + 릴레이 둘 다) — 아니면 ComfyUI 입력
  폴더의 아무 사진이 "생성 후보"가 되어 프로필(얼굴 레퍼런스)로 고를 수 있었다. LoRA 로더는 기본 허용 목록에서 뺐다(텍스트
  하한이 볼 수 없다). 쓸 수 없는 템플릿으로 갈 컷은 실사로 계획했다가 실패하지 않고 처음부터 NovelAI 로 계획된다.
- **얼굴 레퍼런스 기록**: 키가 스튜디오가 저장한 `…/profile_*` 이어야 하고, 기록된 프롬프트에 읽을 수 있는 영어 단어가
  있어야 한다(보이지 않는 문자·기호뿐인 기록은 거절). 한국어로만 친 테스트 프롬프트는 사진 프롬프트에서 빠지므로, 고를 때
  실제로 렌더한 사진 프롬프트를 `photoPrompt` 로 같이 남겨 그것이 이미지를 보증한다. 앱은 참조를 객체 키로만 보낸다.
- **어휘**: 영어 굴절·동의어·완곡어(in the altogether, giving head …), 하이픈/공백 복합어, Danbooru 노출 태그,
  나이 표기(소수·괄호·성별 표기·상한/하한·생년·졸업 연도·학년), 아동 체형 조합(여성 + 아동 체형 서술 2종 이상 + 수영복/속옷),
  라틴 문자 외국어(es/pt/fr/de/it/nl/pl/cs/tr/북유럽/ro/hu/hr/id/tl/vi + 로마자 일·중·한), 암시적 노출(욕조·온천·샤워),
  **노출 목적 의상**(open-crotch, cupless, peephole bra …). 마지막 것은 동봉된 여성의 삶 시드의 기본 의상에서 나왔다 —
  10명 중 9명의 엉덩이가 보이는 컷은 이제 NovelAI 로 고정되고, 상반신·클로즈업 컷은 프레임 필터가 그 옷을 빼므로 실사로 남는다.
- **변장 읽기**(`readFloorVariants`, 하한에만 쓰고 범주 판단에는 안 쓴다): 리트(n4ked, p0rn), 단어 안 기호를 가린 글자로
  (n.ked, p**n, n*a*k*e*d — 검열 단어 사전과 글자 수로 맞춘다), 비슷한 기호(× → x, ∪ → u), 쪼갠 단어(na ke d), 모든 결합
  부호 제거, 문자 같은 기호(🅽, 지역 표시 문자) 매핑. 그림 문자(이모지)는 사진 프롬프트에서 전부 빠지고, 성적·아동 이모지는
  하한 신호가 된다. NFKC 뒤에도 라틴 문자가 아닌 글자가 남은 토큰은 검증할 수 없어 NovelAI 로 고정한다(한글 구간은 사진
  프롬프트에서 빠지므로 판단에서도 빠진다). ø ß ł đ ı œ 같은 분해되지 않는 라틴 글자는 접어서 읽는다.

측정(레드팀 프로브, 사진 프롬프트까지 잠기지 않고 간 비율, 처음 → 지금): 영어 노골 어휘 149/185 → 8, 미성년 코드
103/167 → 5, 라틴 문자 외국어 102/105 → 11, 리트·유니코드 35/119 → 0, Danbooru 노출 태그 106/212 → 52(남은 것은 성인에게
허용되는 선정 태그), 47 페이로드 × 18 경로 행렬 794 칸 → 0, 네거티브 70/75 → 0. **일상 SFW 335개는 처음부터 지금까지 0개
잠김.** 의도적으로 안 잡는 것: 성인에게 허용되는 선정 어휘(lingerie, upskirt, micro bikini …), 일상 뜻이 있는 단어(melons,
little sister, Kind …). 거짓 양성은 NovelAI 로 그려지는 것으로 끝난다(nude heels, weekend in Bath, Sex and the City …).

**한계.** 어휘 하한은 완전할 수 없다 — 레드팀은 매 라운드 새 표현을 찾았다. 이것은 DynamicChat 이 *스스로* 노골·미성년
글을 실사로 만들지 않게 하는 장치이지, ComfyUI 를 직접 돌릴 수 있는 로컬 사용자를 막는 보안 경계가 아니다(릴레이는
직접 호출자의 글을 그대로 넘긴다). 더 강한 다음 단계는 **완성된 이미지를 로컬 비전 모델로 검사**하는 것이다 — 지금 이 PC
에는 비전 모델이 없어 넣지 않았다.

**신체 정체성 특성은 explicit 이 아니다.** `futanari` 같은 특성은 캐릭터가 *누구인지*이지 컷이 *무엇을 보여주는지*가
아니다. 정체성은 모든 컷에 주입되므로, 처음 구현처럼 이것을 explicit 으로 보면 동봉된 성인 시드(여성의 삶)에서 실제
한 턴의 16컷이 전부 NovelAI 로 고정됐다 — 카페에서 옷을 입고 앉아 있는 컷까지. 지금은 `identity-trait:` 신호로만 남고,
컷이 실제로 보여주는 것(노출·성기·행위)은 여전히 explicit 어휘가 잡는다. 대신 이 특성은 **실사 프롬프트에 절대 들어가지
않는다** — 사진은 나머지 캡션이 묘사하는 옷 입은 인물만 그린다.

**실사 프롬프트 정화.** 애니 모델은 양식화하지만 사진 모델은 그대로 그리는 것들: 비침 소재 수식어(see-through /
translucent / sheer)는 떼고 옷은 불투명하게 남긴다("translucent tactical bodysuit" → "tactical bodysuit"), 섭식장애
체형 단어(anorexic, emaciated …)는 버린다. 둘 다 같은 시드의 기본 의상·정체성에서 QA 중에 발견했다.

## 프롬프트

실사 모델의 텍스트 인코더는 LLM(Qwen3-VL / Qwen3)이라 태그 나열보다 문장을 잘 따른다. 그래서 LLM 을 한 번
더 부르지 않고, 이미 합성된 NAI 계획을 **결정적으로** 문장으로 바꾼다(`photorealPrompt.ts`): NAI 전용 토큰
(품질·작가·연도·가중치)을 버리고, 의상은 "wearing …", 표정·동작·장소·조명을 문장으로 묶고, 프레임은 카메라
문장(“Medium close-up from the waist up, eye-level”)이 된다. 캐릭터에 `photorealPrompt`(자연어 외형)가 있으면
NAI 정체성 태그 대신 그 문장이 쓰인다.

**구도가 이긴다.** 사진 모델은 들은 옷을 전부 보여주려고 샷을 넓힌다. NAI 의 프레임 필터는 LLM 이 *작성한* 의상을
의도적으로 남기지만, 실사 프롬프트에서는 크롭이 보여줄 수 없는 의상을 뺀다 — 얼굴 클로즈업 캡션에 청바지가 남아
있으면 3/3 시드가 허리 위 샷으로 나왔다(실측). 카메라 문장은 스타일 리드 바로 다음에 둔다(사람·장소 뒤에 두면 졌다).
머리가 보이는 크롭은 "얼굴이 보인다"고 명시한다 — 기본 스타일의 원천인 데일리룩 사진은 얼굴을 자르거나 휴대폰으로
가리는 구도가 많아서, 명시하지 않으면 스토리 컷에서도 얼굴이 잘렸다. 한 태그에 동작 전체를 쓴 경우("trembling
hands reaching toward a brown leather handle")는 소재 단어가 있어도 옷이 아니라 동작 절로 둔다.

스타일 프리셋의 기본값 `daily_look` 은 사용자의 핀터레스트 저장 핀(샤오홍슈·핀터레스트 데일리룩: 스마트폰
일상 사진, 부드러운 자연광, 크림·화이트 실내, 낮은 대비의 파스텔 보정)에서 왔다. 프리셋의 조명·공간은
**기본값일 뿐** 컷이 말하지 않을 때만 들어간다 — 처음엔 항상 "창가 낮빛"을 붙여서 네온 밤거리 컷과 충돌했다.

## 얼굴 고정 (프로필 이미지 = 레퍼런스)

제작 화면에서 캐릭터마다 NAI / 실사를 골라 테스트 생성하고, 마음에 드는 후보를 **프로필로 지정**하면
`CharacterVisualProfile.profileImages[backend]` 에 저장된다(바이트는 객체 저장소, 상태에는 키만).

- **실사**: Qwen-Image 2.1 은 생성과 다중 레퍼런스 편집을 한 모델로 한다. 프롬프트에서 `<image1>` 로 부르고,
  끝에 "레퍼런스는 정체성만, 배경·의상·포즈·조명은 이 설명에서" 문장을 붙인다.
- **레퍼런스는 머리·어깨로 잘라서 넣는다** (`computeReferenceHeadCrop`, ComfyUI `ImageCrop`). 실측: 프로필
  전체를 넣으면 일부 시드에서 모델이 레퍼런스를 **편집 대상**으로 취급해 프로필의 방과 옷을 그대로 다시 그렸다
  (출력 캔버스가 레퍼런스 크기와 같을 때 가장 심했다). 머리 크롭으로 바꾸자 실패하던 시드 포함 9/9 가 새 장면·
  새 의상으로 이동했고 얼굴은 유지됐다. 베낄 옷과 방이 보이지 않기 때문이다.
- **NovelAI**: Precise Reference(`director_reference_*` 다섯 배열). **V4.5 전용**이고 이미지당 +5 Anlas(Opus
  포함)라 기본은 꺼짐. Vibe Transfer 와 한 요청에 섞을 수 없어서 레퍼런스가 붙으면 그 요청의 vibe 는 빠진다.
  참조 이미지는 1024x1536 / 1536x1024 / 1472x1472 중 하나에 검은 여백으로 맞춰야 하므로 프로필 지정 시점에
  한 번 만들어 둔다(`referenceObjectKey`).

### 제작 화면: 캐릭터 프로필 스튜디오

캐릭터 프롬프트를 고르면 `CharacterProfileImagePanel` 이 열린다.

1. **백엔드** [NovelAI | 실사] 를 고르고 프롬프트를 쓴다(NovelAI: 태그, 기본값은 visual profile + 기본 의상 / 실사: 자연어
   외형, 예: "a Korean woman in her late twenties with shoulder-length wavy brown hair").
2. **테스트 생성**(후보 1~4장). 후보는 컴포넌트 메모리에만 있다 — 초안·앱 상태·localStorage 어디에도 바이트가 들어가지
   않는다. 계획은 `characterProfileStudio.ts` 가 하고 실행은 이야기 컷과 같은 백엔드별 큐와 GPU 잠금을 탄다.
3. **프로필로 지정** — 그 이미지를 객체 저장소에 저장하고(NovelAI 는 레퍼런스용 레터박스 사본도) 기록에 크기를 남긴다.
   이후 그 백엔드의 모든 컷에서 이 얼굴이 레퍼런스가 된다.
4. **얼굴 고정 테스트**(실사) — 프로필을 레퍼런스로 다른 포즈 2장을 그려 레퍼런스가 잘 먹는지 확인한다.

실사 테스트는 로컬이라 무료다. NovelAI 프로필은 Precise Reference 를 켰을 때만 레퍼런스가 된다(V4.5, 이미지당 +5 Anlas).

**지운 프로필 파일 정리.** 바꾸거나 해제한 프로필 이미지(`profile_*`)는 저장할 때 지워지고, 버려진 초안 폴더
(`objects/draft_sim_*/`, 7일 경과)는 서버 시작 때 정리된다. 둘 다 **어떤 시뮬레이션도 그 키를 쓰지 않을 때만**
지운다 — 시뮬레이션 사본은 키를 공유하고, 대기 중인 작업이 레퍼런스로 물고 있을 수도 있다. 이야기 컷 이미지는
이름으로 걸러 절대 건드리지 않는다. 기존 시뮬레이션을 편집하다 버린 후보는 그 시뮬레이션 폴더에 남는다(알려진 한계,
`docs/security-hardening-notes.md`).

## ComfyUI 와 모델 (16GB 기준)

ComfyUI 는 브라우저의 교차 출처 요청을 거부하므로(Origin/Host 검사) 서버 릴레이를 쓴다. `--enable-cors-header`
를 켜라고 안내하지 않는다 — ComfyUI 의 유일한 CSRF 방어를 끄는 것이다.

| 프리셋 | 파일 (models/…) | 라이선스 | 비고 |
|---|---|---|---|
| Qwen-Image 2.1 Turbo (기본) | diffusion_models/qwen_image_2.1_turbo_int8_convrot (7.3GB), text_encoders/qwen3vl_8b_int8_convrot (9.4GB), vae/qwen_image_2.1_vae_bf16 (0.7GB) | Qwen Research (**비상업**) | 생성 + 얼굴 레퍼런스 |
| Z-Image Turbo | diffusion_models/z_image_turbo_int8_convrot (6.2GB), text_encoders/qwen_3_4b_fp8_mixed (5.6GB), vae/ae (0.3GB) | Apache-2.0 | 생성 전용 |
| Qwen-Image 2.1 / Edit 2511 | 기본 설치에 없음 | Qwen Research / Apache-2.0 | 선택 |

실측 (RTX 4080 SUPER 16GB, ComfyUI 0.40.0):

| 작업 | 시간 |
|---|---|
| Qwen 2.1 Turbo 960x1280, 콜드(17GB 적재 포함) / 웜 | 9.0s / 3.3s |
| Z-Image Turbo 960x1280, 콜드(모델 교체) / 웜 | 7.2s / 3.7s |
| 레퍼런스 1장 컷 | 8~15s |

## VRAM 교대 (로컬 LLM ↔ 로컬 실사)

16GB 에는 qwen3:14b(16k)와 실사 모델이 동시에 넉넉히 들어가지 않는다. 실측:

| 상황 | 결과 |
|---|---|
| ComfyUI 가 VRAM 을 쥔 상태에서 LLM 적재 | 첫 호출 34.6s (적재 19.7s, 12.4 tok/s), 이후 44.9 tok/s |
| ComfyUI `/free` 후 LLM | 5.7s (적재 3.8s), 63~65 tok/s |
| LLM 이 VRAM 을 쥔 상태에서 이미지 | 10.8s |
| LLM 언로드 후 이미지 | 3.1s |

그래서 `vramHandoff: "auto"` 일 때(실사 켜짐 + ComfyUI 루프백 + LLM 이 루프백 Ollama) `localGpuCoordinator`
가 LLM 턴과 실사 작업을 하나의 FIFO 잠금으로 직렬화하고, 바뀔 때마다 상대를 내린다: 이미지 전에는 Ollama
모델을 `keep_alive: 0` 으로 언로드, LLM 전에는 ComfyUI `/free`. 턴당 LLM 재적재 4~5초로 이미지가 약 3배 빨라지고
LLM 은 전속력을 유지한다.

**VRAM 은 시뮬레이션이 아니라 기계의 것이다.** 실사 설정은 시뮬레이션마다 다르지만, 이 카드에 모델을 올리는 작업은
전부 잠금을 탄다 — 실사를 끈 시뮬레이션의 로컬 Ollama 턴도, 호스티드 LLM 시뮬레이션의 로컬 렌더도. "지금 상대가
올라와 있나" 로 잠금 여부를 정하던 때에는, 실사를 끈 시뮬레이션의 자동 진행 턴이 잠금 밖에서 돌고 그 사이 제작 화면의
실사 테스트가 빈 잠금을 잡아 턴의 모델을 생성 도중에 내리고 그 옆에서 렌더했다(실측 16.4GB 중 15.0~15.8GB 사용).
QA 에서 본 유일한 ComfyUI 프로세스 사망(CUDA illegal memory access)이 이 겹침 중에 일어났다. 중단(`/interrupt`) 자체는
원인이 아니다 — 웜·콜드 상태, 적재·샘플링·디코드 각 구간에서 36번 중단해 0번 죽었다. 기다리는 상대가 없으면 잠금은
비용이 없다. 시뮬레이션에서 교대를 끈(`vramHandoff: "off"`) 작업만 잠금 밖에서 돈다.

## 릴레이 보안

`/photoreal/comfyui/{generate,status,free,interrupt}` 는 전부 `rejectNonLocalSecretRequest` 뒤에 있다(로컬 GPU 를
쓰고 ComfyUI 입력 폴더에 파일을 쓴다). 그 가드는 순서대로:

1. **Host 검사** — 루프백 이름/IP 가 아니면(또는 `DYNAMICCHAT_ALLOWED_HOSTS` 에 없으면) 403. DNS 리바인딩 페이지는
   Origin 없이 같은 출처 GET 을 보내므로 Origin 만으로는 막을 수 없었다.
2. **ComfyUI 출처 거부** — ComfyUI 포트(8188, `DYNAMICCHAT_COMFYUI_URL` 의 포트, `DYNAMICCHAT_DENIED_ORIGINS` 의 포트)는
   루프백 표기(localhost, 127.x, [::1], ::ffff:127.x …)와 스킴에 상관없이 거부. 커스텀 노드의 프론트엔드 JS 가 그 출처에서 돈다.
3. Origin 검사(기존).

baseUrl 은 루프백만(또는 `DYNAMICCHAT_COMFYUI_HOSTS`), 워크플로 노드는 허용 목록(`DYNAMICCHAT_COMFYUI_EXTRA_NODES`),
레퍼런스는 객체 저장소 경로 가드를 거친다. Vite 개발 서버는 `cors: false` 와 `fs.deny` 에 `.dynamicchat-data` 를 둔다 —
데이터 폴더가 웹 루트 안에 있어서, ComfyUI 출처의 스크립트가 Vite 로 `dev-secrets.json` 을 읽을 수 있었다(실측).

내장 워크플로의 출력은 `SaveImage` 가 아니라 `PreviewImage` 다 — 사진이 ComfyUI 의 `output/` 에 영구히 쌓이지 않고
`temp/`(ComfyUI 재시작 때 비워짐)에 남는다. 레퍼런스 업로드는 내용 주소 이름으로 `input/dynamicchat/` 에 남는다.
자세한 규칙은 `docs/security-hardening-notes.md`.

## 평가

```powershell
npm run eval:photoreal-suite   # 아래 전부 (키·GPU 불필요)
npm run eval:photoreal         # 파이프라인 통합 (mock ComfyUI + 실제 API 서버 자식 프로세스)
npm run eval:photoreal-safety  # 안전 하한 코퍼스: 반드시 잠길 것 + 일상 SFW 335개는 절대 안 잠길 것 + 여성의 삶 컷
npm run eval:profile-objects   # 지운 프로필 파일 정리 (다른 시뮬레이션이 쓰는 키는 절대 안 지움)
npm run eval:gpu-coordinator   # VRAM 교대 순서 (겹침 없음, 첫 턴 전 ComfyUI 해제)
```

**첫 턴 전에도 ComfyUI 를 내린다.** 페이지를 새로 열면 코디네이터는 이전 세션이 카드에 무엇을 남겼는지 모른다. 그래서
첫 이미지 전에 Ollama 를 내리는 것과 대칭으로, 첫 로컬 LLM 턴 전에 그 상태의 ComfyUI 를 한 번 비운다 — 남아 있던
ComfyUI 모델 옆에서 16k qwen3:14b 의 주석 호출이 157초 만에 타임아웃됐다(16.4GB 중 15.6GB 사용, 실측).
