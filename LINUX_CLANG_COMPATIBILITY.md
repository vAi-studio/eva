# Linux C++ 컴파일러 호환성 수정 보고서

## 개요

이 문서는 EVA 라이브러리를 **Linux + Clang 15 + libc++** 및 **Linux + GCC 13 + libstdc++** 환경에서 빌드할 수 있도록 수정한 내용을 정리합니다.

**문제 원인**: Windows MSVC에서는 정상 빌드되지만, Linux에서 C++20/23 표준 준수가 더 엄격하여 컴파일 에러가 발생했습니다.

---

## 수정 내역

### 1. `emplace_back` → `push_back` + Designated Initializer

**문제**: C++20부터 `std::construct_at`이 aggregate 초기화를 지원하지 않아, C 스타일 구조체(Vulkan 타입)에 `emplace_back`이 동작하지 않음.

**해결**: `push_back` + designated initializer 사용

**수정 위치** (`eva-runtime.cpp`):
- VkDescriptorSetLayoutBinding (line ~3114)
- VkDescriptorBufferInfo (line ~3344)
- VkDescriptorImageInfo (line ~3364)
- VkMemoryBarrier2, VkBufferMemoryBarrier2, VkImageMemoryBarrier2 (line ~1794, ~1805, ~1862)
- VkMemoryBarrier, VkBufferMemoryBarrier (line ~1935, ~1945)
- VkDeviceQueueCreateInfo (line ~1059)
- VkSemaphoreSubmitInfo (line ~1402, ~1433)
- VkCommandBufferSubmitInfo (line ~1420)

**예시**:
```cpp
// Before (Windows MSVC OK, Linux FAIL)
vkBindings.emplace_back(binding, type, count, flags, nullptr);

// After (Cross-platform)
vkBindings.push_back(VkDescriptorSetLayoutBinding{
    .binding = binding,
    .descriptorType = type,
    .descriptorCount = count,
    .stageFlags = flags,
    .pImmutableSamplers = nullptr
}); // TODO: Linux 호환성 - emplace_back → push_back + designated initializer
```

---

### 2. Forward Declaration → Complete Type

**문제**: `std::vector<T>` 기본값 초기화에서 `T`가 complete type이어야 함.

**수정 위치** (`eva-runtime.h`):
- `CopyRegion` 구조체를 forward declaration 위치로 이동 (line ~139)

---

### 3. 템플릿 함수 정의 위치 이동

**문제**: 템플릿 함수가 반환 타입의 complete type 정의 전에 위치.

**수정 위치** (`eva-runtime.h`):
- `Device::createSemaphores<N>()` 선언만 남기고 구현을 `Semaphore` 클래스 정의 뒤로 이동 (line ~505)
- `DescriptorPool::operator()` 템플릿 구현을 `DescriptorSet` 정의 뒤로 이동 (line ~650)

---

### 4. `static_assert(false)` → Dependent False

**문제**: C++23에서 `if constexpr` 내부의 `static_assert(false)`가 항상 평가됨.

**수정 위치** (`eva-runtime.cpp`):
```cpp
// Before
static_assert(false, "Invalid VkResource type");

// After
static_assert(sizeof(VkResource) == 0, "Invalid VkResource type");
```

---

### 5. 템플릿 파라미터 타입 불일치

**문제**: `ConstantID`는 `uint32_t ID`인데 `constant_id` 함수와 `operator+`는 `int ID`로 정의.

**수정 위치** (`eva-runtime.h`):
- `constant_id<int ID>` → `constant_id<uint32_t ID>` (line ~891, ~902)
- `ShaderStage::operator+<int ID>` → `operator+<uint32_t ID>` (line ~994, ~1005)

---

### 6. BindingInfo 생성자 추가

**문제**: libc++의 `std::map::try_emplace`가 aggregate의 piecewise 생성을 지원하지 않음.

**수정 위치** (`eva-runtime.h`):
```cpp
struct BindingInfo {
    // ... 멤버들 ...

    // 생성자 추가
    BindingInfo() = default;
    BindingInfo(uint32_t b, DESCRIPTOR_TYPE d, uint32_t c, SHADER_STAGE s)
        : binding(b), descriptorType(d), descriptorCount(c), stageFlags(s) {}
};
```

---

### 7. `inline` 함수 정의 위치

**문제**: `.cpp` 파일에 `inline` 키워드가 있는 함수는 다른 번역 단위에서 링킹 에러 발생 (GCC Release 빌드에서 발견).

**수정 위치** (`eva-runtime.cpp`):
```cpp
// Before (GCC Release 빌드에서 undefined reference 에러)
inline uint64_t Buffer::size() const
{
    return impl().size;
}

// After
uint64_t Buffer::size() const
{
    return impl().size;
}
```

**참고**: 헤더에 인라인으로 이동하려면 `Impl` 타입이 complete type이어야 하므로, `.cpp`에서 `inline` 제거가 더 간단한 해결책입니다.

---

### 8. `<cmath>` 헤더 누락

**문제**: GCC에서 `powf`, `sinf` 등 수학 함수가 암시적으로 포함되지 않음.

**수정 위치** (`vai-builtin-nodes.cpp`):
```cpp
#include <cmath>  // powf, sinf 사용을 위해 추가
```

---

## 빌드 설정

### build.sh

Debug 빌드 시 libc++ 사용:
```bash
EXTRA_CXX_FLAGS="-stdlib=libc++"
EXTRA_LINKER_FLAGS="-stdlib=libc++ -lc++abi"
```

Release 빌드 시 GCC + libstdc++ 사용 (기본값).

### 필요 패키지 (Ubuntu)

```bash
# Clang + libc++ (Debug 빌드용)
sudo apt install clang-15 libc++-15-dev libc++abi-15-dev

# GCC (Release 빌드용)
sudo apt install g++-13
```

---

## 검색용 주석

모든 수정 위치에 다음 주석이 포함되어 있습니다:
```
TODO: Linux 호환성
```

grep으로 검색 가능:
```bash
grep -rn "TODO: Linux" external/eva/
```

---

## 테스트 환경

- OS: Ubuntu 22.04
- Compiler (Debug): Clang 15.0.7 + libc++ 15
- Compiler (Release): GCC 13.1.0 + libstdc++
- C++ Standard: C++23 (gnu++2b / gnu++23)
- Vulkan SDK: 1.3.290.0

---

## 향후 권장사항

1. **Windows에서 테스트**: 이 수정들은 MSVC에서도 동작해야 하지만 검증 필요
2. **Clang 18+ 고려**: 최신 Clang은 GCC libstdc++와 호환성이 개선됨
3. **libc++ 기본 사용 고려**: Cross-platform 일관성을 위해 모든 플랫폼에서 libc++ 사용 검토
