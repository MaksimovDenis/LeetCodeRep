package tree

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// time: O(N), memory: O(h) or O(N)
func preorderTraversal(root *TreeNode) []int {
	var result []int
	travers(root, &result)
	return result
}

func travers(root *TreeNode, result *[]int) {
	if root == nil {
		return
	}

	*result = append(*result, root.Val)

	travers(root.Left, result)
	travers(root.Right, result)
}

func inorderTraversal(root *TreeNode) []int {
	var result []int
	intravers(root, &result)
	return result
}

func intravers(root *TreeNode, result *[]int) {
	if root == nil {
		return
	}

	intravers(root.Left, result)
	*result = append(*result, root.Val)
	intravers(root.Right, result)
}

func postorderTraversal(root *TreeNode) []int {
	var result []int
	postTravers(root, &result)
	return result
}

func postTravers(root *TreeNode, result *[]int) {
	if root == nil {
		return
	}

	postTravers(root.Left, result)
	postTravers(root.Right, result)
	*result = append(*result, root.Val)
}

func rightSideView(root *TreeNode) []int {
    var result []int
    rightSideViewTravers(root, 0, &result)
    return result
}

func rightSideViewTravers(root *TreeNode, level int, result *[]int) {
    if root == nil {
        return
    }

    if len(*result) <= level {
        *result = append(*result, 0)
    }

    (*result)[level] = root.Val
    rightSideViewTravers(root.Left, level+1, result)
    rightSideViewTravers(root.Right, level+1, result)
}