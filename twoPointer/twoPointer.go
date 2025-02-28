package twopointer

import "strings"

func SquareOfASortedArray(nums []int) []int {
	var result []int

	left := 0
	right := len(nums) - 1

	for left <= right {
		leftVal := nums[left] * nums[left]
		rightVal := nums[right] * nums[right]

		if leftVal < rightVal {
			result = append(result, leftVal)
			left++
		} else {
			result = append(result, rightVal)
			right--
		}
	}

	return result
}

func MoveZeroes(nums []int) {
	if len(nums) < 2 {
		return
	}

	valPointer := 0
	wherePointer := 0

	for wherePointer < len(nums) {
		if nums[valPointer] == 0 {
			valPointer++
		} else {
			nums[wherePointer] = nums[valPointer]
			wherePointer++
			valPointer++
		}
	}

	for i := wherePointer; i < len(nums); i++ {
		nums[i] = 0
	}
}

func isPalindrome(s string) bool {
	left := 0
	right := len(s) - 1

	for left < right {
		for left < right && !isAlphanumeric(s[left]) {
			left++
		}

		for left < right && !isAlphanumeric(s[right]) {
			right--
		}

		if strings.ToLower(string(s[left])) != strings.ToLower(string(s[right])) {
			return false
		}
		left++
		right--
	}

	return true
}

func isAlphanumeric(c byte) bool {
	return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
}

// Two Sum II - Input Array Is Sorted
func twoSum(numbers []int, target int) []int {
	left := 0
	right := len(numbers) - 1

	result := make([]int, 0, 2)

	for left < right {
		val := numbers[left] + numbers[right]

		switch {
		case val == target:
			result = append(result, left+1, right+1)
			return result
		case val < target:
			left++
		case val > target:
			right--
		}
	}

	return result
}
