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

// --------SAME TREE---------------------------
// time: O(N) mem: O(2h)
func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil || q == nil {
		return p == q
	}

	if p.Val != q.Val {
		return false
	}

	return isSameTree(q.Left, p.Left) && isSameTree(q.Right, p.Right)
}

// -------PathSum------------------------------
// time: O(N) mem: O(h)
func hasPathSum(root *TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}

	return hasPathSumHelper(root, targetSum, 0)
}

func hasPathSumHelper(root *TreeNode, targetSum int, count int) bool {
	if root == nil {
		return false
	}

	count += root.Val

	if targetSum == count && isLeaf(root) {
		return true
	}

	return hasPathSumHelper(root.Left, targetSum, count) || hasPathSumHelper(root.Right, targetSum, count)
}

func isLeaf(root *TreeNode) bool {
	return root.Left == nil && root.Right == nil
}
