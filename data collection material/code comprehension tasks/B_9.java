package com.drawing;

import java.util.ArrayList;
import java.util.List;

public class B_9 {

      public static void main(String[] args) {

        List<Object> array = new ArrayList<Object>();

        Circle a = new Circle();
        Triangle b = new Triangle();
        Square c = new Square();
        Triangle d = new Triangle();

        array.add(a);
        array.add(b);
        array.add(c);
        array.add(d);

        for(int i=array.size()-1; i>=0; i--){
            Graphics.draw(array.get(i));
        }

    }
}


/*
 *
 * What is the sequence of shape objects drawn by Main()?
 *
 */

