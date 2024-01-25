package main

import (
	"fmt"
	"sort"
)

func main() {

	fmt.Println(isValid("[][]"))

}

func twoSumLong(nums []int, target int) []int {
	var a []int
mainLoop:
	for i := 0; i < len(nums); i++ {
		for j := i + 1; j < len(nums); j++ {
			sum := nums[i] + nums[j]
			if sum == target {
				a = append(a, i, j)
				break mainLoop
			}
		}
	}
	return a
}

func twoSumShort(nums []int, target int) []int {
	sortedNums := make([][2]int, len(nums))

	for i, num := range nums {
		sortedNums[i] = [2]int{i, num}
	}

	sort.Slice(sortedNums, func(i, j int) bool {
		return sortedNums[i][1] < sortedNums[j][1]
	})

	left, right := 0, len(nums)-1

	for left < right {
		currentSum := sortedNums[left][1] + sortedNums[right][1]

		if currentSum == target {
			return []int{sortedNums[left][0], sortedNums[right][0]}
		} else if currentSum < target {
			left++
		} else {
			right--
		}
	}

	return []int{}
}

func isPalindrome(x int) bool {
	if x < 0 {
		return false
	}
	num := x
	reserved := 0
	for x != 0 {
		reserved = reserved*10 + x%10
		x = x / 10
	}
	return reserved == num
}

func romanToInt(s string) int {
	var a []int
	for i := 0; i < len(s); i++ {
		switch {
		case string(s[i]) == "I":
			a = append(a, 1)
		case string(s[i]) == "V":
			a = append(a, 5)
		case string(s[i]) == "X":
			a = append(a, 10)
		case string(s[i]) == "L":
			a = append(a, 50)
		case string(s[i]) == "C":
			a = append(a, 100)
		case string(s[i]) == "D":
			a = append(a, 500)
		case string(s[i]) == "M":
			a = append(a, 1000)
		}
	}
	var max int
	sum := 0
	for i := len(a) - 1; i >= 0; i-- {
		if a[i] < max {
			sum -= a[i]
		} else {
			sum += a[i]
		}
		max = a[i]
	}
	return sum
}

func longestCommonPrefix(strs []string) string {
	if len(strs) == 1 { // handle only 1 element
		return strs[0]
	}

	// sort them first, the most different one will be in first and last
	sort.Strings(strs)

	// compare first and last
	l := len(strs)
	for i := range strs[0] {
		fmt.Println(i)
		if strs[0][i] != strs[l-1][i] {
			return strs[0][:i]
		}
	}
	return strs[0]
}

func isValid(s string) bool {
	var a bool
mainLoop:
	for i := 0; i < len(s)-1; i++ {
		if len(s)%2 != 0 {
			a = false
			break mainLoop
		}
		switch {
		case string(s[i]) == "(":
			if string(s[i+1]) == ")" {
				a = true
			} else {
				a = false
				break mainLoop
			}
		case string(s[i]) == "[":
			if string(s[i+1]) == "]" {
				a = true
			} else {
				a = false
				break mainLoop
			}
		case string(s[i]) == "{":
			if string(s[i+1]) == "}" {
				a = true
			} else {
				a = false
				break mainLoop
			}
		}
	}
	return a
}
