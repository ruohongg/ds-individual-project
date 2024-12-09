def example_function(x: int, y: float, z: str, w: list):
    result = 0
    if x > 100:
        result = y * x
        if "alpha" in z:
            result += len(z) ** 2
        elif "beta" in z:
            result -= len(z) ** 3
        else:
            result *= len(z)
        if len(w) > 5:
            result += sum(w[:5]) / len(w)
    elif x < 0:
        result = sum(w) * y
        if all(i < 0 for i in w):
            result -= abs(x) * 2
        elif any(i % 2 == 0 for i in w):
            result += sum(i ** 2 for i in w if i % 2 == 0)
    else:
        result = y / (x + 2)
        if z.isalpha():
            result += ord(z[0]) * len(w)
        elif z.isdigit():
            result *= int(z) + len(w)
        elif len(z) > 10:
            result -= len(z) ** 0.5

    if len(w) >= 5:
        for i in range(len(w) - 4):
            result += sum(w[i:i + 5])
    elif len(w) == 0:
        result -= 100
    else:
        for i in range(len(w)):
            if w[i] > 10:
                result += w[i] * 2
            else:
                result -= w[i] ** 0.5

    if z.startswith("data"):
        result *= len(z) ** 1.5
    elif z.endswith("end"):
        result /= (len(z) or 1)
    else:
        vowels = {"a", "e", "i", "o", "u"}
        vowel_count = sum(1 for char in z if char.lower() in vowels)
        consonant_count = len(z) - vowel_count
        result += vowel_count * 2 - consonant_count

    if x > 50 and len(w) > 3:
        for i in range(1, len(w)):
            result += w[i] * i
            if w[i] > 20:
                result -= x / w[i]
            if "special" in z:
                result *= len(z)
    elif x <= 10:
        for i, val in enumerate(w):
            result += (val * i) % x if x != 0 else 1
            if i % 2 == 0 and x > 5:
                result += 10
            elif i % 2 == 1 and len(z) > 5:
                result -= len(w)
    if x == 0:
        result += sum(w) or 1
    if not z:
        result -= 50
    if not w:
        result += 100

    if x % 5 == 0:
        result += y ** 2
    if len(w) % 2 == 0:
        result -= sum(w[:len(w)//2]) * 0.1

    sorted_w = sorted(w, reverse=True)
    for i in range(min(10, len(sorted_w))):
        result += sorted_w[i] * (i + 1)

    if z.lower() == "final":
        result /= 3
    elif "debug" in z:
        result *= 1.2

    return result