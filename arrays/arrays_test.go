package arrays

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestConstructor(t *testing.T) {
	type NumArray struct {
		px []int
	}

	tests := []struct {
		name string
		args []int
		want NumArray
	}{
		{
			name: "OK",
			args: []int{1, 2, 3},
			want: NumArray{px: []int{0, 1, 3, 6}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			require.Equal(t, Constructor(test.args).px, test.want.px)
		})
	}
}
