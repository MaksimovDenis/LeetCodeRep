package main

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"
)

func main() {
	l1 := &ListNodeTwo{Val: 2, Next: &ListNodeTwo{Val: 4, Next: &ListNodeTwo{Val: 3}}}

	l2 := &ListNodeTwo{Val: 5, Next: &ListNodeTwo{Val: 6, Next: &ListNodeTwo{Val: 4}}}
	list := addTwoNumbers(l1, l2)
	displayList(list)

}

type ListNode1 struct {
	Val  int
	Next *ListNode1
}

func reserveList(head *ListNode1) *ListNode1 {
	return helper(head, nil)
}

func helper(current *ListNode1, prev *ListNode1) *ListNode1 {
	if current != nil {
		return prev
	}
	next := current.Next
	current.Next = prev
	return helper(next, current)
}

func yandexVasyaMasha() {}
func yandexGolas()      {}
func yandexPresses()    {}
func yandexChess()      {}
func yandexFindProfit() {}
func yandexMath()       {}
func yandexSchedule()   {}
func yandexLines()      {}
func yandexFish()       {}
func yandexCoordintes() {}

func RLE(s string) string {
	var counter int = 1
	var result string
	for i := 1; i < len(s); i++ {
		if s[i] == s[i-1] {
			counter++
		} else {
			if counter > 1 {
				result += string(s[i-1]) + strconv.Itoa(counter)
				counter = 1
			} else {
				result += string(s[i-1])
			}
		}
	}
	if counter > 1 {
		result += string(s[len(s)-1]) + strconv.Itoa(counter)
	} else {
		result += string(s[len(s)-1])
	}
	return result
}

func containsNearbyDuplicate2(nums []int, k int) bool {

	hashMap := make(map[int]int)
	for i, v := range nums {
		if val, ok := hashMap[v]; ok {
			if i-val <= k {
				return true
			}
		}
		hashMap[v] = i
	}
	return false
}

func containsNearbyAlmostDuplicate(nums []int, indexDiff int, valueDiff int) bool {

	hashMap := make(map[int][]int)
	for i := 0; i < len(nums); i++ {
		hashMap[nums[i]] = append(hashMap[nums[i]], i)
		if len(hashMap[nums[i]]) > 1 {
			ar := hashMap[nums[i]]
			for j := 0; j < len(ar); j++ {
				if (j + 1) == (len(ar) - 1) {
					difIndex := math.Abs(float64(ar[j] - ar[j+1]))
					difVal := math.Abs(float64(nums[ar[j]] - nums[ar[j+1]]))
					if (int(difIndex) <= indexDiff) && (i != j) && (int(difVal) <= valueDiff) {
						return true
					} else {
						break
					}
				} else {
					difIndex := math.Abs(float64(ar[j] - ar[j+1]))
					difVal := math.Abs(float64(nums[ar[j]] - nums[ar[j+1]]))
					if (int(difIndex) <= indexDiff) && (i != j) && (int(difVal) <= valueDiff) {
						return true
					}
				}
			}
		}
	}
	return false
}

func containsDuplicate(nums []int) bool {
	hashMap := make(map[int]bool)
	for i := 0; i < len(nums); i++ {
		if hashMap[nums[i]] {
			return true
		} else {
			hashMap[nums[i]] = true
		}
	}
	return false
}

func majorityElement2(nums []int) int {
	var element, count int
	for i := 0; i < len(nums); i++ {
		switch {
		case count == 0:
			element = nums[i]
			count++
		case element == nums[i]:
			count++
		default:
			count--
		}
	}
	return element
}

func majorityElement(nums []int) int {
	hashMap := make(map[int]int)
	max := 0
	for i := 0; i < len(nums); i++ {
		hashMap[nums[i]] += 1
		if hashMap[nums[i]] > max {
			max = i
		}
	}
	return nums[max]
}

func generate(numRows int) [][]int {
	array := make([][]int, numRows)
	for i := 0; i < numRows; i++ {
		array[i] = make([]int, i+1)
		array[i][0] = 1
		array[i][i] = 1
		for j := 1; j < i; j++ {
			array[i][j] = array[i-1][j-1] + array[i-1][j]
		}
	}
	return array
}

func getRow(rowIndex int) []int {
	array := make([][]int, rowIndex+1)
	var result []int
	if rowIndex == 0 {
		result = []int{1}
		return result
	} else if rowIndex == 1 {
		result = []int{1, 1}
		return result
	} else {
		for i := 0; i <= rowIndex; i++ {
			array[i] = make([]int, i+1)
			array[i][0] = 1
			array[i][i] = 1
			for j := 1; j < i; j++ {
				array[i][j] = array[i-1][j-1] + array[i-1][j]
			}
		}
		return array[rowIndex]
	}
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

func isValid1(s string) bool {
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

func countNodes(root *TreeNode) int {
	var counter int
	var inorder func(node *TreeNode)
	inorder = func(root *TreeNode) {
		if root != nil {
			counter += 1
			inorder(root.Left)
			inorder(root.Right)
		}
	}
	inorder(root)
	return counter
}

func intersection(a, b []int) []int {
	var intersectionArray []int
	m := make(map[int]int)

	for v := range a {
		if _, ok := m[v]; !ok {
			m[v] = 1
		} else {
			m[v] += 1
		}
	}

	for v := range b {
		if _, ok := m[v]; ok && m[v] > 0 {
			intersectionArray = append(intersectionArray, v)
			m[v] -= 1
		}
	}
	return intersectionArray
}

func minDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	left := minDepth(root.Left)
	right := minDepth(root.Right)
	if left == 0 || right == 0 {
		return 1 + left + right
	} else {
		return min(left, right)
	}
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
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

func (root *BinaryTree) Insert(value int) {
	if root.Root == nil {
		root.Root = &TreeNode{Val: value, Left: nil, Right: nil}
	} else {
		InsertRecursive(root.Root, value)
	}
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

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	if list1 == nil {
		return list2
	}
	if list2 == nil {
		return list1
	}
	if list1.Val < list2.Val {
		list1.Next = mergeTwoLists(list1.Next, list2)
		return list1
	} else {
		list2.Next = mergeTwoLists(list1, list2.Next)
		return list2
	}
}

func mergeTwoLists2(list1 *ListNode, list2 *ListNode) *ListNode {
	dummy := &ListNode{}
	list := dummy
	var tempVal int
	for list1 != nil && list2 != nil {
		if list1.Val < list2.Val {
			tempVal = list1.Val
			list1 = list1.Next
		} else {
			tempVal = list2.Val
			list2 = list2.Next
		}
		list.Next = &ListNode{Val: tempVal}
		list = list.Next
	}
	if list1 == nil {
		list.Next = list1
	}
	if list2 == nil {
		list.Next = list2
	}
	return dummy.Next
}

/*func singleNumber(nums []int) int {
	value := nums[0]
	for i := 1; i < len(nums); i++ {
		if (nums[i] == value) || (nums[len(nums)-1] == value) {
			if (i+1) != (len(nums)) && (nums[i+1]) != value {
				value = nums[i+1]
				i++
			} else if (i+1) != (len(nums)) && (nums[i]) != value {
				value = nums[i]
			}
		} else {
			continue
		}
	}
	return value

	vanya, ok := hashmap["vanya_identity"]
}*/

func singleNumber(nums []int) int {
	mapValues := make(map[int]int)
	for i := 0; i < len(nums); i++ {
		mapValues[nums[i]] = mapValues[nums[i]] + 1
	}
	for k, v := range mapValues {
		if v == 1 {
			return k
		}
	}
	return 0
}

func singleNumberXOR(nums []int) int {
	result := 0
	for i := 0; i < len(nums); i++ {
		result ^= nums[i]
	}
	return result
}

func preorderTraversal(root *TreeNode) []int {
	var ar []int
	var inorder func(*TreeNode)
	inorder = func(root *TreeNode) {
		if root != nil {
			ar = append(ar, root.Val)
			inorder(root.Left)
			inorder(root.Right)
		}
	}
	inorder(root)
	return ar
}

func getDecimalValue(head *ListNode) int {
	var number int
	var helper func(*ListNode)
	helper = func(head *ListNode) {
		if head != nil {
			//Сдвигаем число на 1 бит (если первыое число отчно от нуля то добавляем следующий бит)
			number = (number << 1) | head.Val
			helper(head.Next)
		}
	}
	helper(head)

	return number
}

func middleNode(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}
	return slow
}

func newReverse(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}

	var revHead *ListNode

	for head != nil {
		tmp := head.Next
		head.Next = revHead
		revHead = head
		head = tmp
	}
	return revHead
}

func isPalindromeList(head *ListNode) bool {

	if head == nil {
		return false
	}

	var array []int

	for head != nil {
		array = append(array, head.Val)
		head = head.Next
	}

	for i := 0; i < len(array)/2; i++ {
		if array[i] != array[len(array)-1-i] {
			return false
		}
	}

	return true
}

func SortZeroAndOne(arr []int) {
	left := 0
	right := len(arr) - 1

	for left < right {
		for left == 0 && left < right {
			left++
		}

		for right == 1 && left < right {
			right--
		}

		for left < right {
			arr[left], arr[right] = arr[right], arr[left]
			left--
			right++
		}
	}
}

func postorderTraversal(root *TreeNode) []int {
	var arr []int
	var inorder func(*TreeNode)
	inorder = func(root *TreeNode) {
		if root != nil {
			inorder(root.Left)
			inorder(root.Right)
			arr = append(arr, root.Val)
		}
	}
	inorder(root)
	return arr
}

func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	root := &TreeNode{Val: nums[len(nums)/2]}
	root.Left = sortedArrayToBST(nums[:len(nums)/2])
	root.Right = sortedArrayToBST(nums[len(nums)/2+1:])
	return root
}

func InsertRecursive(root *TreeNode, value int) {
	if root.Val > value {
		if root.Left == nil {
			root.Left = &TreeNode{Val: value, Left: nil, Right: nil}
		}
		InsertRecursive(root.Left, value)
	}
	if root.Val < value {
		if root.Right == nil {
			root.Right = &TreeNode{Val: value, Left: nil, Right: nil}
		}
		InsertRecursive(root.Right, value)
	}
}

func Stone(a, b string) int {
	hashMap := make(map[rune]struct{})

	for _, v := range a {
		hashMap[v] = struct{}{}
	}
	var result int
	for _, v := range b {
		if _, ok := hashMap[v]; ok {
			result++
		}
	}
	return result
}

func summaryRanges(nums []int) []string {

	var result []string

	if len(nums) == 0 {
		return result
	}

	start := nums[0]
	prev := nums[0]

	for _, val := range nums[1:] {
		if val-prev > 1 {
			if start == prev {
				result = append(result, fmt.Sprintf("%d", start))
			} else {
				result = append(result, fmt.Sprintf("%d->%d", start, prev))
			}
			start = val
		}
		prev = val
	}

	if start == prev {
		result = append(result, fmt.Sprintf("%d", start))
	} else {
		result = append(result, fmt.Sprintf("%d->%d", start, prev))
	}

	return result
}

func joinChannels(chs ...<-chan int) <-chan int {
	multiplexChannel := make(chan int)

	var wg sync.WaitGroup

	multiplex := func(c <-chan int) {
		defer wg.Done()
		for i := range c {
			multiplexChannel <- i
		}
	}

	wg.Add(len(chs))

	for _, c := range chs {
		go multiplex(c)
	}

	go func() {
		wg.Wait()
		close(multiplexChannel)
	}()

	return multiplexChannel

}

/*type MyStack struct {
	queue []int
}

func Constructor() MyStack {
	return MyStack{}
}

func (this *MyStack) Push(x int) {
	this.queue = append(this.queue, x)
}

func (this *MyStack) Pop() int {
	tmp := this.queue[len(this.queue)-1]
	this.queue = this.queue[:len(this.queue)-1]
	return tmp
}

func (this *MyStack) Top() int {
	return this.queue[len(this.queue)-1]
}

func (this *MyStack) Empty() bool {
	if len(this.queue) == 0 {
		return true
	}
	return false
}*/

func invertTree(root *TreeNode) *TreeNode {
	var inorder func(root *TreeNode)
	inorder = func(root *TreeNode) {
		if root != nil {
			tmp := root.Left
			root.Left = root.Right
			root.Right = tmp
			inorder(root.Left)
			inorder(root.Right)
		}
	}
	inorder(root)
	return root
}

func addDigits(num int) int {
	if num < 10 {
		return num
	}
	var helper func(num int) int
	helper = func(num int) int {
		var value int
		numStr := strconv.Itoa(num)
		for _, v := range numStr {
			tmp, _ := strconv.Atoi(string(v))
			value = value + tmp
		}
		num = value
		if num < 10 {
			return num
		}
		return helper(num)
	}
	return helper(num)
}

type MyQueue struct {
	stack []int
}

func Constructor() MyQueue {
	return MyQueue{}
}

func (this *MyQueue) Push(x int) {
	this.stack = append(this.stack, x)
}

func (this *MyQueue) Pop() int {
	tmp := this.stack[0]
	this.stack = this.stack[1:len(this.stack)]
	return tmp
}

func (this *MyQueue) Peek() int {
	return this.stack[0]
}

func (this *MyQueue) Empty() bool {
	if len(this.stack) == 0 {
		return true
	}
	return false
}

func isAnagram(s string, t string) bool {
	if len(s) == len(t) {
		str := make(map[string]int)
		for i := 0; i < len(s); i++ {
			str[string(s[i])] += 1
			str[string(t[i])] -= 1
		}
		for _, v := range str {
			if v != 0 {
				return false
			}
		}
		return true
	}
	return false
}

func missingNumber(nums []int) int {
	sort.Ints(nums)
	var result int
	if nums[0] != 0 {
		return 0
	}
	for i := 0; i < len(nums)-1; i++ {
		if nums[i]+1 == nums[i+1] {
			continue
		} else {
			result = nums[i] + 1
			return result
		}
	}
	return nums[len(nums)-1] + 1
}

func moveZeroes(nums []int) {
	sort.Ints(nums)
	if len(nums) > 1 {
		count := 0
		for i := 0; nums[i] == 0; i++ {
			count++
		}
		r := 0
		for i := 0; i < count; i++ {
			nums = append(nums, 0)
			nums = nums[r+1:]

		}
	}
	fmt.Println(nums)
}

func wordPattern(pattern string, s string) bool {
	var tmp string
	newS := strings.Split(s, " ")
	if len(string(pattern)) != len(newS) {
		return false
	}

	strMap := make(map[rune]string)
	used := make(map[string]bool)
	for i, v := range pattern {
		if _, ok := strMap[v]; !ok {
			if tmp != newS[i] && used[newS[i]] == false {
				strMap[v] = newS[i]
				tmp = newS[i]
				used[newS[i]] = true
				continue
			} else {
				return false
			}
		} else {
			if strMap[v] != newS[i] {
				return false
			}
		}
	}
	return true

}

func wordPattern1(pattern string, s string) bool {
	newS := strings.Split(s, " ")
	if len(string(pattern)) != len(newS) {
		return false
	}

	strMap := make(map[string]string)
	for i := 0; i < len(newS); i++ {
		if _, ok := strMap[newS[i]]; !ok {
			strMap[newS[i]] = string(pattern[i])
			continue
		} else {
			if strMap[newS[i]] != string(pattern[i]) {
				return false
			}
		}
	}
	return true

}

func canWinNim(n int) bool {
	return n%4 != 0
}

func isPowerOfThree(n int) bool {
	if n < 1 {
		return false
	}
	for n%3 == 0 {
		n = n / 3
	}

	return n == 1
}

func countBits(n int) []int {
	var result []int
	var sum int
	for i := 0; i <= n; i++ {
		x_double := strconv.FormatInt(int64(i), 2)
		for _, j := range x_double {
			number, _ := strconv.Atoi(string(j))
			sum += number
		}
		result = append(result, sum)
		sum = 0
	}
	return result
}

func reverseString(s []byte) {
	for i := 0; i < len(s)/2; i++ {
		s[i], s[len(s)-1-i] = s[len(s)-1-i], s[i]
	}
}

func isPowerOfFour(n int) bool {
	if n < 1 {
		return false
	}
	for n%4 == 0 {
		n = n / 4
	}
	return n == 1
}

func reverseVowels(s string) string {

	str := make(map[string]struct{})
	str["a"] = struct{}{}
	str["e"] = struct{}{}
	str["i"] = struct{}{}
	str["o"] = struct{}{}
	str["u"] = struct{}{}

	var slice []string
	for _, i := range s {
		slice = append(slice, string(i))
	}

	left := 0
	right := len(slice) - 1

	for left < right {

		_, leftIsVowel := str[strings.ToLower(slice[left])]

		for left < right && !leftIsVowel {
			left++
			_, leftIsVowel = str[strings.ToLower(slice[left])]
		}

		_, rightIsVowel := str[strings.ToLower(slice[right])]

		for left < right && !rightIsVowel {
			right--
			_, rightIsVowel = str[strings.ToLower(slice[right])]
		}

		for left < right {
			slice[left], slice[right] = slice[right], slice[left]
			left++
			right--
			break
		}
	}
	result := strings.Join(slice, "")
	return result
}

func intersectionArrays(nums1 []int, nums2 []int) []int {

	compare := make(map[int]struct{})
	var result []int

	if len(nums1) >= len(nums2) {
		for i := 0; i < len(nums1); i++ {
			compare[nums1[i]] = struct{}{}
		}
		for i := 0; i < len(nums2); i++ {
			if _, ok := compare[nums2[i]]; ok {
				result = append(result, nums2[i])
				delete(compare, nums2[i])
			}
		}
	} else {
		for i := 0; i < len(nums2); i++ {
			compare[nums2[i]] = struct{}{}
		}
		for i := 0; i < len(nums1); i++ {
			if _, ok := compare[nums1[i]]; ok {
				result = append(result, nums1[i])
				delete(compare, nums1[i])
			}
		}
	}
	return result
}

func intersectArray2(nums1 []int, nums2 []int) []int {

	var result []int
	array := make(map[int]int)

	if len(nums1) >= len(nums2) {
		for i := 0; i < len(nums2); i++ {
			array[nums2[i]] += 1
		}
		for i := 0; i < len(nums1); i++ {
			if array[nums1[i]] > 0 {
				result = append(result, nums1[i])
				array[nums1[i]] -= 1
			}
		}
	} else {
		for i := 0; i < len(nums1); i++ {
			array[nums1[i]] += 1
		}
		for i := 0; i < len(nums2); i++ {
			if array[nums2[i]] > 0 {
				result = append(result, nums2[i])
				array[nums2[i]] -= 1
			}
		}
	}
	return result
}

func isPerfectSquare(num int) bool {
	var i, n int
	for n < num {
		n = i * i
		i++
	}
	return n == num
}

func canConstruct(ransomNote string, magazine string) bool {

	arr := make(map[rune]int)

	for _, v := range magazine {
		arr[v] += 1
	}
	for _, v := range ransomNote {
		if arr[v] == 0 {
			return false
		}
		arr[v] -= 1
	}
	return true
}

func firstUniqChar(s string) int {

	arr := make(map[rune]int)

	for _, v := range s {
		arr[v] += 1
	}
	for i, v := range s {
		if arr[v] == 1 {
			return i
		}
	}
	return -1
}

func findTheDifference(s string, t string) byte {
	var result byte
	arr := make(map[byte]int)

	if len(s) == 0 {
		result = byte(t[len(t)-1])
	}

	for _, v := range t {
		arr[byte(v)] += 1
	}
	for _, v := range s {
		arr[byte(v)] -= 1
		if arr[byte(v)] == 0 {
			delete(arr, byte(v))
		}
	}

	for key := range arr {
		result = key
	}
	return result
}

func isSubsequence(s string, t string) bool {

	var arr []string
	var count int
	if len(s) == 0 {
		return true
	}
	for _, v := range s {
		arr = append(arr, string(v))
	}

	for i := 0; i < len(string(t)); i++ {
		if arr[count] == string(t[i]) {
			count++
			if count == len(string(s)) {
				return true
			}
		}
	}
	return false
}

func sumOfLeftLeaves(root *TreeNode) int {
	var result int
	var inorder func(*TreeNode)
	inorder = func(root *TreeNode) {
		if root != nil {
			if root.Left != nil && root.Left.Left == nil && root.Left.Right == nil {
				result += root.Left.Val
			}
			inorder(root.Left)
			inorder(root.Right)
		}
	}
	inorder(root)
	return result
}

func longestPalindrome(s string) int {
	arr := make(map[rune]int)
	var result int
	var check bool

	for _, v := range s {
		arr[v] += 1
	}

	for _, v := range arr {
		if v%2 == 0 {
			result += v
		} else if v%2 != 0 {
			result += v - 1
			check = true
		}
	}

	if check {
		result++
	}

	return result
}

func fizzBuzz(n int) []string {

	var result []string
	for i := 0; i < n; i++ {
		switch {
		case (i+1)%3 == 0 && (i+1)%5 == 0:
			result = append(result, "FizzBuzz")
		case (i+1)%3 == 0 && (i+1)%5 != 0:
			result = append(result, "Fizz")
		case (i+1)%5 == 0:
			result = append(result, "Buzz")
		default:
			result = append(result, strconv.Itoa(i+1))
		}
	}
	return result
}

func thirdMax(nums []int) int {
	sort.Ints(nums)
	arr := make(map[int]struct{})
	var result []int

	for i := 0; i < len(nums); i++ {
		if _, ok := arr[nums[i]]; !ok {
			result = append(result, nums[i])
			arr[nums[i]] = struct{}{}
		}
	}
	if len(result) >= 3 {
		return result[len(result)-3]
	} else if len(result) == 2 {
		return result[len(result)-1]
	} else {
		return result[len(result)-1]
	}
}

func countSegments(s string) int {

	var str string
	var count int = 1

	for _, v := range s {
		if string(v) != " " {
			str += string(v)
		} else {
			strings.TrimSpace(str)
			if str != "" {
				count += 1
				str = ""
			}
		}
	}
	if str != "" {
		return count
	}
	return count - 1
}

func arrangeCoins(n int) int {

	var count int
	r := n
	if n == 1 {
		return 1
	}
	for i := 0; i < n; i++ {
		r = r - i
		if r > 0 {
			count++
		} else if r == 0 {
			return count
		} else if r < 0 {
			break
		}
	}
	return count - 1
}

func findDisappearedNumbers(nums []int) []int {
	sort.Ints(nums)
	var arr []int
	for i := 0; i < len(nums); i++ {
		if i != len(nums)-1 && nums[i+1] != nums[i]+1 && nums[i+1] != nums[i] {
			arr = append(arr, nums[i]+1)
		}
	}
	if len(arr) > 2 {
		arr = append(arr[:1], arr[len(arr)-1:]...)
	}
	return arr
}

func isPowerOfTwo(n int) bool {
	if n < 1 {
		return false
	}
	for n%2 == 0 {
		n = n / 2
	}
	return n == 1
}

func repeatedSubstringPattern(s string) bool {
	doubled := s + s
	return strings.Contains(doubled[1:len(doubled)-1], s)
}

func twoSum(nums []int, target int) []int {
	hash := make(map[int]int)

	for index, value := range nums {
		complement := target - value
		if i, ok := hash[complement]; ok {
			return []int{i, index}
		}
		hash[value] = index
	}
	return []int{}
}

func isValid(s string) bool {
	cont := make(map[string]int)

	for _, v := range s {
		cont[string(v)]++
	}

	if cont["{"] != cont["}"] {
		return false
	} else if cont["("] != cont[")"] {
		return false
	} else if cont["["] != cont["]"] {
		return false
	}

	return true
}

type ListNodeTwo struct {
	Val  int
	Next *ListNodeTwo
}

func addTwoNumbers(l1 *ListNodeTwo, l2 *ListNodeTwo) *ListNodeTwo {
	var num1, num2 int
	var helper1, helper2 func(l *ListNodeTwo, num int)
	helper1 = func(l *ListNodeTwo, num1 int) {
		if l != nil {
			num1 = num1*10 + l.Val
			l = l.Next
			helper1(l, num1)
		}
	}

	helper1(l1, num1)

	helper2 = func(l *ListNodeTwo, num int) {
		if l != nil {
			num = num*10 + l.Val
			l = l.Next
			helper2(l, num)
		}
		return
	}
	helper2(l2, num2)

	sum := num1 + num2

	var l3 *ListNodeTwo

	for sum != 0 {
		newNode := &ListNodeTwo{Val: sum % 10}
		sum /= 10
		l3.Next = newNode
	}
	return l3
}

func displayList(head *ListNodeTwo) {
	current := head
	for current != nil {
		fmt.Print(current.Val, " ")
		current = current.Next
	}
	fmt.Println()
}

// Time complexity: Constructor: O(1)
//
//	Add: O(1)
//	Remove: O(1)
//	Contains: O(1)
type MyHashSet struct {
	count         int
	filledBuckets int
	bucketCount   int
	buckets       [][]int
	loadFactor    int
}

func ConstructorHash() MyHashSet {
	return MyHashSet{
		count:         0,
		filledBuckets: 0,
		bucketCount:   8,
		buckets:       make([][]int, 8),
		loadFactor:    80,
	}
}

func (this *MyHashSet) hashFunc(key int) int {
	// mask last x bits to find corresponding bucket
	return key & (this.bucketCount - 1)
}

func (this *MyHashSet) Add(key int) {
	hash := this.hashFunc(key)
	// if already exists, do nothing
	for _, val := range this.buckets[hash] {
		if val == key {
			return
		}
	}
	// 80% of filled buckets = bucketCount * 80 / 100
	// if at load capacity, add buckets and redistribute
	if this.filledBuckets >= this.bucketCount*this.loadFactor/100 {
		this.bucketCount = this.bucketCount * 2
		temp := this.buckets
		this.buckets = make([][]int, this.bucketCount)
		this.filledBuckets = 0
		for _, bucket := range temp {
			for _, val := range bucket {
				newHash := this.hashFunc(val)
				if this.buckets[newHash] == nil {
					this.filledBuckets++
				}
				this.buckets[newHash] = append(this.buckets[newHash], val)
			}
		}
	}
	hash = this.hashFunc(key)
	if this.buckets[hash] == nil {
		// an empty bucket will be filled
		this.filledBuckets++
	}
	this.buckets[hash] = append(this.buckets[hash], key)
	this.count++
}

func (this *MyHashSet) Remove(key int) {
	hash := this.hashFunc(key)
	for i, val := range this.buckets[hash] {
		if val == key {
			this.buckets[hash] = append(this.buckets[hash][:i],
				this.buckets[hash][i+1:]...)
			this.count--
			if len(this.buckets[hash]) == 0 {
				this.filledBuckets--
			}
		}
	}
}

func (this *MyHashSet) Contains(key int) bool {
	hash := this.hashFunc(key)
	for _, val := range this.buckets[hash] {
		if val == key {
			return true
		}
	}
	return false
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {

	m := make(map[*ListNode]bool)

	var helper func(list *ListNode)
	helper = func(list *ListNode) {
		m[list] = true
		if list.Next != nil {
			helper(list.Next)
		}
	}
	if headA != nil {
		helper(headA)
	}
	var result *ListNode
	helper = func(list *ListNode) {
		if _, ok := m[list]; ok {
			result = list
		} else {
			if list.Next != nil {
				helper(list.Next)
			}
		}
	}
	if headB != nil {
		helper(headB)
	}
	return result
}

func getIntersectionNodeV2(headA, headB *ListNode) *ListNode {
	seen := make(map[*ListNode]bool)

	for n := headA; n != nil; n = n.Next {
		seen[n] = true
	}

	for n := headB; n != nil; n = n.Next {
		if _, ok := seen[n]; ok {
			return n
		}
	}

	return nil
}
