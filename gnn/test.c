#include <stdio.h>
#include <stdint.h>

int main() {
    int num1 = 5;
    int num2 = 10;
    int sum;

    sum = num1+num2;

    printf("The sum is: %d\n", sum);

    if (sum > 10) {
        printf("Sum is greater than 10.\n");
    } else {
        printf("Sum is 10 or less.\n");
    }

    for (int i = 0; i < 5; i++) {
        printf("%d ", i);
    }

    return 0;
}
