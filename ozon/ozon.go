package ozon

func a() {
    x := []int{}
    x = append(x, 0) 
    x = append(x, 1) 
    x = append(x, 2) 
    y := append(x, 3) 
    z := append(x, 4)
    fmt.Println(y, z) 
}

func main() {
    a()
}



func main() {
    println(handle())
}

type 

func handle() error {
    
}


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
