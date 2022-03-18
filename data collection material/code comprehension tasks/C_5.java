package com.drawing;

import java.util.ArrayList;
import java.util.List;

public class C_5 {

    public static void main(String[] args) {
        List<Object> array = new ArrayList<Object>();

        Object r = new Rectangle();
        Object e = new Triangle();
        Object s = new Circle();

        array.add(r);
        array.add(e);
        array.add(s);
        array.add(s);

        int c=1;
        while(c<=array.size()){
            int nextIndex = next(c,array);
            Graphics.draw(array.get(nextIndex));
            c++;
        }

    }

    public static int next(int c, List<Object> array){
        return Math.abs((array.size()/2) -c);
    }


}
/*
 *
 * What is the sequence of shape objects drawn by Main()?
 *
 */
