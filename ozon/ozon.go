package ozon

import (
	"fmt"
)

func a() {
	x := []int{}
	x = append(x, 0)  // 0
	x = append(x, 1)  // 0 1
	x = append(x, 2)  // 0 1 2
	y := append(x, 3) // 0 1 2 3
	z := append(x, 4) // 0 1 2 4
	fmt.Println(y, z) // 0 1 2 4 | 0 1 2 4
}

func handle() error {
	return &newErr{errString: "что-то пошло не так"}
}

func main() {
	println(handle())
}

type newErr struct {
	errString string
}

func (me *newErr) Error() string {
	return me.errString
}

/*
func main() {
	var wg sync.WaitGroup

	for i := 0; i < 5; i++ {
		wg.Add(1)

		go func(i int) {
			defer wg.Done()
			fmt.Println(i)
		}(i)
	}

	wg.Wait()
}
*/
