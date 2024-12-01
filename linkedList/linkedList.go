package linkedlist

type ListNode struct {
	Value int
	Next  *ListNode
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummyNode := &ListNode{Next: head}

	var length int

	curr := dummyNode

	for curr.Next != nil {
		length++
		curr = curr.Next
	}

	curr = dummyNode

	for i := 0; i <= (length - n - 1); i++ {
		curr = curr.Next
	}

	curr.Next = curr.Next.Next

	return dummyNode.Next
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

	slow := head
	fast := head

	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}

	return slow
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
