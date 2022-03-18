package com.drawing;

import java.util.ArrayList;
import java.util.List;

public class C_3 {

    public static void main(String[] args) {
        List<Object> array = new ArrayList<Object>();

        Object o = new Rectangle();
        array.add(o);
        o = new Square();
        array.add(o);
        o = new Rectangle();
        array.add(o);
        o = new Triangle();
        array.add(o);
        o = new Circle();
        array.add(o);


        int i = 0;
        while(i<array.size()){
            if(i!=array.size()-1){
                Graphics.draw(array.get(i));
            }
            else {
                int nextIndex = index(111);
                Graphics.draw(array.get(nextIndex));
            }
            i++;
        }
    }

    public static int index(int n) {
        if (n == 0) {
            return 0;
        }
        return (n % 10) + index((int) n/10);
    }


}
/*
 *
 * What is the sequence of shape objects drawn by Main()?
 *
 */