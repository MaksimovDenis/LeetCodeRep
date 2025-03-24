package yandextask

type ListNode struct {
	Val  int
	Next *ListNode
}

func reverseList(head *ListNode) *ListNode {
	curr := head
	var prev *ListNode

	for curr != nil {
		tmp := curr
		curr = curr.Next
		tmp.Next = prev
		prev = tmp
	}

	return prev
}

// 1. Ищем середину списка
// 2. Выполняем реверс 2ой половины списка
// 3. Сравниваем значением обои половин списка
func isPalindrome(head *ListNode) bool {
	// middle of the linked list
	fast := head
	slow := head

	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}

	// Reverse
	curr := slow
	var prev *ListNode

	for curr != nil {
		tmp := curr
		curr = curr.Next
		tmp.Next = prev
		prev = tmp
	}

	p1 := head
	p2 := prev

	// compare two lists
	for p1 != nil && p2 != nil {
		if p1.Val != p2.Val {
			return false
		}

		p1 = p1.Next
		p2 = p2.Next
	}

	return true
}

// 1. Пишем функцию для слияния 2х листов
// 2. Пишем функцию для слияних общего списка
func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) == 0 {
		return nil
	}

	if len(lists) > 1 {
		var mergeLists []*ListNode

		for i := 0; i < len(lists); i += 2 {
			l1 := lists[i]

			var l2 *ListNode
			if (i + 1) < len(lists) {
				l2 = lists[i+1]
			}

			mergeLists = append(mergeLists, mergeTwoSortedLists(l1, l2))
		}

		lists = mergeLists
	}

	return lists[0]
}

func mergeTwoSortedLists(list1, list2 *ListNode) *ListNode {
	if list1 == nil {
		return list2
	}

	if list2 == nil {
		return list1
	}

	dummy := &ListNode{}
	list := dummy
	var tmpVal int

	for list1 != nil && list2 != nil {
		if list1.Val < list2.Val {
			tmpVal = list1.Val
			list1 = list1.Next
		} else {
			tmpVal = list2.Val
			list2 = list2.Next
		}

		list.Next = &ListNode{Val: tmpVal}
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

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}

	if l2 == nil {
		return l1
	}

	dummy := &ListNode{}
	list := dummy
	var curr int

	for l1 != nil || l2 != nil || curr != 0 {
		sum := 0
		if l1 != nil {
			sum += l1.Val
			l1 = l1.Next
		}

		if l2 != nil {
			sum += l2.Val
			l2 = l2.Next
		}

		sum += curr
		list.Next = &ListNode{Val: sum % 10}
		list = list.Next
		curr = sum / 10
	}

	return dummy.Next
}

func hasCycle(head *ListNode) bool {
	slow := head
	fast := head

	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next

		if fast == slow {
			return true
		}
	}

	return false
}
