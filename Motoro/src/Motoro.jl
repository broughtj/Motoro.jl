module Motoro

function game() 
    secret_number = rand(1:100)
    tries = 0
    guess = 0

    println("I'm thinking of a number between 1 and 100.")
    println("Can you guess it?")

    while guess != secret_number
        print("Your guess: ")
        guess = parse(Int64, readline())
        tries += 1

        if guess < secret_number
            println("Too low. Try again.")
        elseif guess > secret_number
            println("Too high. Try again.")
        else 
            println("Congratulations. You got it!")
        end 
    end
end 

end # module Motoro
