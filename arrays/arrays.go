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
