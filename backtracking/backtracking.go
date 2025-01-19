package backtracking

func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return []string{}
	}

	//Таблица цифр и их комбинаций
	phoneMap := map[byte]string{
		'2': "abc",
		'3': "def",
		'4': "ghi",
		'5': "jkl",
		'6': "mno",
		'7': "pqrs",
		'8': "tuv",
		'9': "wxyz",
	}

	output := []string{}

	var backtrack func(combination string, nextDigits string)
	backtrack = func(combination string, nextDigits string) {
		// Случай, когда больше нет цифр для проверки
		if len(nextDigits) == 0 {
			output = append(output, combination)
			return
		}

		// Берём цифру и значения для неё
		currentDigit := nextDigits[0]
		letters := phoneMap[currentDigit]

		for i := 0; i < len(letters); i++ {
			backtrack(combination+string(letters[i]), nextDigits[1:])
		}
	}

	// Начинаем бэктрегинг с нулвыми комбинциями и всеми цифрами
	backtrack("", digits)
	return output
}
