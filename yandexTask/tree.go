package yandextask

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// -------LEVEL ORDER-----------------------
// time: O(n) mem: O(n+h) = O(2N)
func levelOrder(root *TreeNode) [][]int {
	var result [][]int
	preOrdereLevelOrder(root, 0, &result)
	return result
}

func preOrdereLevelOrder(root *TreeNode, level int, result *[][]int) {
	if root == nil {
		return
	}

	if len(*result) <= level {
		*result = append(*result, []int{})
	}

	(*result)[level] = append((*result)[level], root.Val)
	level = level + 1
	preOrdereLevelOrder(root.Left, level, result)
	preOrdereLevelOrder(root.Right, level, result)
}

// -------SYMMETRICAL TREE-----------------------
func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}

	return helperSymmetrical(root.Left, root.Right)
}

func helperSymmetrical(left, right *TreeNode) bool {
	if left == nil || right == nil {
		return left == right
	}

	if left.Val != right.Val {
		return false
	}

	return helperSymmetrical(left.Left, right.Right) && helperSymmetrical(left.Right, right.Left)
}
