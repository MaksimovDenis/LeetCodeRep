package arrays

import (
	"math/rand"
	"strconv"
)

type NumArray struct {
	px []int
}

func Constructor(nums []int) NumArray {
	var px []int
	px = append(px, 0)

	for i := 0; i < len(nums); i++ {
		px = append(px, px[i]+nums[i])
	}

	return NumArray{px}
}

func (this *NumArray) SumRange(left int, right int) int {
	return this.px[right+1] - this.px[left]
}

func pivotIndex(nums []int) int {
	var sum, currSum int

	for _, value := range nums {
		sum += value
	}

	for i := 0; i < len(nums); i++ {
		temp := sum - currSum - nums[i]

		if temp == currSum {
			return i
		}

		currSum += nums[i]
	}

	return -1
}

func subarraySum(nums []int, k int) int {
	var currPx int
	var count int
	px := make(map[int]int)

	px[0] = 1

	for i := 0; i < len(nums); i++ {
		currPx += nums[i]

		if value, ok := px[currPx-k]; ok {
			count += value
		}

		px[currPx] += 1
	}

	return count
}

func missingNumber(nums []int) int {
	neededSum := ((0 + len(nums) + 1) * len(nums)) / 2
	actualSum := 0

	for _, value := range nums {
		actualSum += value
	}

	return neededSum - actualSum
}

func findDuplicate(nums []int) int {
	seen := make(map[int]bool)
	for _, num := range nums {
		if seen[num] {
			return num
		}
		seen[num] = true
	}
	return -1
}

func rotateArray(nums []int) []int {
	var n []int
	return n
}

func isMonotonic(nums []int) bool {
	if len(nums) < 2 {
		return true
	}

	incr := true
	decr := true

	for i := 1; i < len(nums); i++ {
		incr = incr && nums[i] >= nums[i-1]
		decr = decr && nums[i] <= nums[i-1]
	}

	return incr || decr
}

func findLengthOfLCIS(nums []int) int {
	if len(nums) < 2 {
		return 1
	}

	count := 1
	tmp := 1

	for i := 1; i < len(nums); i++ {
		if nums[i] > nums[i-1] {
			tmp++
		} else {
			if count < tmp {
				count = tmp
			}

			tmp = 1
		}
	}

	if tmp > count {
		count = tmp
	}

	return count
}

func binarySearch(arr []int, target int) int {
	if len(arr) == 0 {
		return -1
	}

	left := 0
	right := len(arr) - 1

	for left <= right {
		mid := (left + right) / 2
		val := arr[mid]

		if val == target {
			return mid
		} else if target < val {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}

	return -1
}

func merge(nums1 []int, m int, nums2 []int, n int) {
	k := m + n - 1
	i := m - 1
	j := n - 1

	for j >= 0 {
		if i >= 0 && nums1[i] > nums2[j] {
			nums1[k] = nums1[i]
			i--
		} else {
			nums1[k] = nums2[j]
			j--
		}
		k--
	}
}

func searchMatrix(matrix [][]int, target int) bool {
	if len(matrix[0]) == 0 || len(matrix) == 0 {
		return false
	}

	for _, m := range matrix {
		if target >= m[0] && target <= m[len(m)-1] {
			if binarySearch2DMatrix(m, 0, len(m)-1, target) {
				return true
			}
		}
	}

	return false
}

func binarySearch2DMatrix(nums []int, left int, right int, tg int) bool {
	for left <= right {
		mid := left + (right-left)/2
		val := nums[mid]

		if tg == val {
			return true
		} else if val < tg {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}

	return false
}

// OZON1
func generate(n, m int) [][]int {
	res := make([][]int, n)
	u := make(map[int]struct{})

	for i := 0; i < n; i++ {
		row := make([]int, m)
		res[i] = row

		for j := 0; j < m; j++ {
			for {
				v := rand.Int()

				if _, ok := u[v]; ok {
					row[j] = v
					u[v] = struct{}{}
					break
				}
			}
		}
	}

	return res
}

func mergeArrs(arrs ...[]int) []int {
	length := 0
	for _, arr := range arrs {
		length += len(arr)
	}

	res := make([]int, length)

	h := res[0:length]

	for _, arr := range arrs {
		copy(h, arr)
		h = h[len(arr):]
	}

	return res
}

var aplphabet = []byte{'x', 'y', 'z'}

func decode(hash string) (password string) {
	for length := 1; ; length++ {
		for i := 0; ; i++ {

			p := strconv.FormatInt(int64(i), len(aplphabet))
			pwd := ""

			for _, c := range p {
				switch c {
				case '0':
					pwd += "x"
				case '1':
					pwd += "y"
				case '2':
					pwd += "z"
				}
			}

			delta := length - len(pwd)
			if delta > 0 {
				for j := 0; j < delta; j++ {
					pwd = "x" + pwd
				}
			}

			if delta < 0 {
				break
			}

			test := dummyhash(pwd)
			if test == hash {
				return pwd
			}

		}
	}
}

func dummyhash(str string) string {
	return str
}
