package linkedlist

type ListNode struct {
	Value int
	Next  *ListNode
}

// Паттерн дамминоуд
// Вычисляем длину -> итерирумеся на len(list) - n -1
// Присваевыаем curr.Next = curr.Next.Next
// сложность O(n) memory = O(n)
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Value: 0, Next: head}

	currList := dummy

	var length int

	for currList != nil {
		length++
		currList = currList.Next
	}

	currList = dummy

	for i := 0; i < length-n-1; i++ {
		currList = currList.Next
	}

	currList.Next = currList.Next.Next

	return dummy.Next
}

// Паттерн дамминоуд
// Берём 2 указателя, сначала проходимся fast на n + 1
// После идём slow до тех пор пока fast != nil
// сложность O(n) memory = O(1)
func removeNthFromEnd2(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Value: 0, Next: head}

	fast := dummy
	slow := dummy

	for i := 0; i < n+1; i++ {
		fast = fast.Next
	}

	for fast != nil {
		slow = slow.Next
		fast = fast.Next
	}

	slow.Next = slow.Next.Next

	return dummy.Next
}

func ReverseList(head *ListNode) *ListNode {
	var prev *ListNode

	curr := head

	for curr != nil {
		tmp := curr
		curr = curr.Next
		tmp.Next = prev
		prev = tmp
	}

	return prev
}

func MiddleOfTheList(head *ListNode) *ListNode {
	if head == nil {
		return head
	}

	fast := head
	slow := head

	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}

	return slow.Next
}

func PalindromLinkedList(head *ListNode) bool {
	if head == nil {
		return true
	}

	slow := head
	fast := head

	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}

	var prev *ListNode

	curr := slow

	for curr != nil {
		tmp := curr
		curr = curr.Next
		tmp.Next = prev
		prev = tmp
	}

	first := head
	second := prev

	for second != nil {
		if first.Value != prev.Value {
			return false
		}

		first = first.Next
		second = second.Next
	}

	return true
}

func reorderList(head *ListNode) {
	if head == nil {
		return
	}

	slow := head
	fast := head

	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}

	second := slow.Next
	slow.Next = nil

	var prev *ListNode

	curr := second

	for curr != nil {
		tmp := curr
		curr = curr.Next
		tmp.Next = prev
		prev = tmp
	}

	second = prev

	first := head

	for second != nil {
		firstNext := first.Next
		secondNext := second.Next

		first.Next = second
		second.Next = firstNext

		first = firstNext
		second = secondNext
	}
}

func PremidNode(head *ListNode) *ListNode {
	slow := head
	fast := head

	for fast != nil && fast.Next != nil && fast.Next.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}

	return slow
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	dummy := &ListNode{}
	list := dummy
	var tmpVal int

	for list1 != nil && list2 != nil {
		if list1.Value < list2.Value {
			tmpVal = list1.Value
			list1 = list1.Next
		} else {
			tmpVal = list2.Value
			list2 = list2.Next
		}

		list.Next = &ListNode{Value: tmpVal}
		list = list.Next
	}

	if list1 == nil {
		list.Next = list2
	}

	if list2 == nil {
		list.Next = list1
	}

	return dummy.Next
}

// input: arr1 = [1, 2, 3], arr2 = [4, 5, 6]
// output: [0, 5, 7, 9] (один ведущий 0 в ответе допускается)
// Note: 123 + 456 = 579

func sumArrays(arr1 []int, arr2 []int) []int {
	switch {
	case len(arr1) == 0 && len(arr2) == 0:
		return arr1
	case len(arr1) == 0:
		return arr2
	case len(arr2) == 0:
		return arr1
	}

	maxLenght := len(max(arr1, arr2)) + 1

	res := make([]int, maxLenght)

	var curr int

	for i := 0; i < len(res); i++ {
		sum := 0
		if i < len(arr1) {
			sum += arr1[len(arr1)-1-i]
		}

		if i < len(arr2) {
			sum += arr2[len(arr2)-1-i]
		}

		sum += curr

		res[len(res)-1-i] = sum % 10

		curr = sum / 10
	}

	return res
}

func max(arr1 []int, arr2 []int) []int {
	if len(arr1) > len(arr2) {
		return arr1
	}

	return arr2
}
