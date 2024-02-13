package main

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

func main() {
	nums := []int{-10, -3, 0, 5, 9}
	PrintInOrder(sortedArrayToBST(nums))

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

func removeDuplicates(nums []int) int {
	i := 0
	for j := range nums {
		if nums[i] != nums[j] {
			i += 1
			nums[i] = nums[j]
		}
	}
	return i + 1
}

func removeElement(nums []int, val int) int {
	i := 0
	for _, v := range nums {
		if v != val {
			nums[i] = v
			i++
		}
	}
	return i
}

func strStr(haystack string, needle string) int {
	switch strings.Contains(haystack, needle) {
	case true:
		return 0
	default:
		return 1
	}
}

func strStrFast(haystack string, needle string) int {
	for i := 0; i <= len(haystack)-len(needle); i++ {
		if haystack[i:len(needle)+1] == needle {
			return 0
		}
	}
	return -1
}

func searchInsert(nums []int, target int) int {
	var l = 0
	var r = len(nums) - 1
	for l <= r {
		midpointer := int((l + r) / 2)
		midValue := nums[midpointer]

		if midValue == target {
			return midpointer
		} else if midValue < target {
			l = midpointer + 1
		} else {
			r = midpointer - 1
		}
	}
	return l
}

func BinarySearch(nums []int, target int) int {
	l := 0
	r := len(nums) - 1
	for l <= r {
		m := l + (r+l)/2
		v := nums[m]

		if v == target {
			return m
		} else if v < target {
			l = m + 1
		} else {
			r = m - 1
		}
	}
	return -1
}

func lengthOfLastWord(s string) int {
	str := strings.Trim(s, " ")
	strSlice := strings.Split(str, " ")
	lastWord := strSlice[len(strSlice)-1]
	return len([]rune(lastWord))
}

func plusOne(digits []int) []int {
	if digits[len(digits)-1] < 9 {
		digits[len(digits)-1] = digits[len(digits)-1] + 1
	} else {
		count := 1
		for i := len(digits) - 1; digits[i] == 9; i-- {
			digits[i] = 0
			if i == 0 {
				break
			}
			count++

		}
		digits[len(digits)-count] = digits[len(digits)-count] + 1
		if digits[0] == 1 {
			digits = append(digits, 0)
		}
	}
	return digits
}

func addBinary(a string, b string) string {
	aInt, _ := strconv.ParseInt(a, 2, 64)
	bInt, _ := strconv.ParseInt(b, 2, 64)
	fmt.Println(aInt, bInt)
	sum := int64(aInt + bInt)
	sumStr := strconv.FormatInt(sum, 2)
	return sumStr
}

func mySqrt(x int) int {
	var i int
	for i = 1; (i * i) <= (x); i += 1 {
		if i*i == (x) {
			return i
		}
		if i*i > (x) {
			i = i - 1
			return i
		}
	}
	return i
}

type ListNode struct {
	Val  int
	Next *ListNode
}

type List struct {
	size int
	head *ListNode
}

func deleteDuplicates(head *ListNode) *ListNode {
	current := head
	for current != nil && current.Next != nil {
		if current.Val == current.Next.Val {
			current.Next = current.Next.Next
			continue
		}
		current = current.Next
	}
	return head
}

func merge(nums1 []int, m int, nums2 []int, n int) {
	nums1 = append(nums1[:m], nums2...)
	sort.Ints(nums1)
}

func maxProfit(prices []int) int {
	min := prices[0]
	profit := 0
	for i := 0; i < len(prices); i++ {
		if prices[i] < min {
			min = prices[i]
		} else if prices[i]-min > profit {
			profit = prices[i] - min
		}
	}
	return profit
}

func isPalindromeN(s string) bool {
	var newString string
	s = strings.ToLower(s)
	for _, v := range s {
		if unicode.IsLetter(v) || (unicode.IsDigit(v)) {
			newString += string(v)
		}
	}
	newStringRune := []rune(newString)
	for i := 0; i < len(newStringRune)/2; i++ {
		if newStringRune[i] != newStringRune[len(newStringRune)-1-i] {
			return false
		}
	}
	return true
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type BinaryTree struct {
	Root *TreeNode
}

func inorderTraversal(root *TreeNode) []int {
	var ar []int
	var inorder func(*TreeNode)
	inorder = func(root *TreeNode) {
		if root != nil {
			inorder(root.Left)
			ar = append(ar, root.Val)
			inorder(root.Right)
		}
	}
	inorder(root)
	return ar
}

func isSameTree(p *TreeNode, q *TreeNode) bool {
	if (p != nil) && (q != nil) {
		return (p.Val == q.Val) && (isSameTree(p.Left, q.Left)) && (isSameTree(p.Right, q.Right))
	} else {
		return p == q
	}
}

func isSymmetrical(root *TreeNode) bool {
	var check func(left, right *TreeNode) bool
	check = func(left, right *TreeNode) bool {
		if left == nil && right == nil {
			return true
		}
		if left == nil || right == nil {
			return false
		}
		if left.Val == right.Val {
			return check(left.Left, right.Right) && check(left.Right, right.Left)
		}
		return false
	}
	return check(root.Left, root.Right)
}

func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return 1 + max(maxDepth(root.Left), maxDepth(root.Right))
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func sortedArrayToBST(nums []int) *TreeNode {
	root := BinaryTree{}
	for i := 0; i < len(nums); i++ {
		root.Insert(nums[i])
	}
	return root.Root
}

func (root *BinaryTree) Insert(value int) {
	if root.Root == nil {
		root.Root = &TreeNode{Val: value, Left: nil, Right: nil}
	} else {
		InsertRecursive(root.Root, value)
	}
}

func InsertRecursive(root *TreeNode, value int) error {
	if root.Val > value {
		if root.Left == nil {
			root.Left = &TreeNode{Val: value, Left: nil, Right: nil}
			return nil
		}
		return InsertRecursive(root.Left, value)
	}
	if root.Val < value {
		if root.Right == nil {
			root.Right = &TreeNode{Val: value, Left: nil, Right: nil}
			return nil
		}
		return InsertRecursive(root.Right, value)
	}
	return nil
}

func (r *BinaryTree) PrintInorderMethod() {
	PrintInOrder(r.Root)
}

func PrintInOrder(value *TreeNode) {
	if value != nil {
		PrintInOrder(value.Left)
		fmt.Print(value.Val, " ")
		PrintInOrder(value.Right)
	}
}
