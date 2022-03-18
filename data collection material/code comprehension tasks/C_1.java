package com.drawing;

import java.util.ArrayList;
import java.util.List;

public class C_1 {

    public static void main(String[] args) {
        List<Object> array = new ArrayList<Object>();

        Object r = new Circle();
        Object i = new Triangle();
        Object v = new Rectangle();
        Object l = new Rectangle();
        Object m = new Circle();
        Object k = new Triangle();


        array.add(r);
        array.add(i);
        array.add(v);
        array.add(l);
        array.add(m);
        array.add(k);


        Graphics.draw(array.get(0));
        Graphics.draw(array.get(1));
        Graphics.draw(array.get(index(3,9)));
        Graphics.draw(array.get(2));
    }


    public static int index(int number1, int number2) {
        int temp = number1;

        while (temp != 0) {
            if (number1 < number2) {
                temp = number1;
                number1 = number2;
                number2 = temp;
            }
            temp = number1 % number2;
            if (temp != 0) {
                number1 = number2;
                number2 = temp;
            }
        }

        return number2;
    }

}


/*
 *
 * What is the sequence of shape objects drawn by Main()?
 *
 */