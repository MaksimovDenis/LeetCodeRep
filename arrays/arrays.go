package arrays

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
