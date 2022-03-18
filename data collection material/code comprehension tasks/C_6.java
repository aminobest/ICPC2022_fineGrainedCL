package com.drawing;

import java.util.ArrayList;
import java.util.List;

public class C_6 {

    public static void main(String[] args) {
        List<Object> array = new ArrayList<Object>();

        Object o = new Object();

        for(int i=0; i<4; i++) {
            if(array.size()%2==0) {
                o = new Circle();
                array.add(o);
                o = new Triangle();
                array.add(o);
            }
            else {
                o = new Rectangle();
                array.add(o);
            }
        }


        int c = 0;
        while (c < array.size()) {
            Graphics.draw(array.get(c));
            c = c + 1;
        }
    }


}
/*
 *
 * What is the sequence of shape objects drawn by Main()?
 *
 */