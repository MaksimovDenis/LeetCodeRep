package linkedlist

type ListNode struct {
	Val  int
	Next *ListNode
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

func reverseList(head *ListNode) *ListNode {
	var prev *ListNode

	cur := head

	for cur != nil {
		tmp := cur
		cur = cur.Next
		tmp.Next = prev
		prev = tmp
	}

	return prev
}
