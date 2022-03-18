package com.drawing;

import java.util.ArrayList;
import java.util.List;

public class B_11 {

    public static void main(String[] args) {
        List<Object> array = new ArrayList<Object>();

        int temp1 = 21;
        int temp2 = 11;

        array.add(new Triangle());
        array.add(new Square());
        array.add(new Triangle());
        Object o = (17 >= temp1) ? ((temp2 > 17) ? new Triangle() : new Square()) : ((temp1 < temp2) ? new Circle() : new Square());
        array.add(o);

        for (int i=1; i<4; i++) {
            Graphics.draw(array.get(i));
        }
    }

}

/*
 *
 * What are the last three shape objects drawn by Main()?
 *
 * (a) triangle, square, triangle
 * (b) circle, square, circle
 * (c) square, triangle, triangle
 * (d) square, triangle, square
 * (e) square, triangle, circle
 *
 */



