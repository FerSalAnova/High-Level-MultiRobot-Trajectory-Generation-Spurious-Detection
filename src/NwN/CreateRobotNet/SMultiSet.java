// A little class for multisets of strings
// Formal specification states that coefficients correspond to the number
// of times an element is in the multiset and, therefore, cannot be negatives

// To compile all the required files, just execute "javac *.java"

import java.util.Set;
import java.util.Iterator;
import java.util.HashMap;
import java.util.HashSet;

public class SMultiSet {
    //val["a"] = 23
    private HashMap<String,Integer> val;

    final static String SEP = "'";

    //---------------------------------------------------------------------------
    // "3'a,2'b,c,4'd"
    // We assume all the coeff. are non-negative
    public SMultiSet(String ms) {
        String[] comps = ms.split(",");

        val = new HashMap<String,Integer>();

        for(int i=0; i<comps.length; i++) {
            String[] term = comps[i].split(SEP);

            if (term.length == 1) { //no coefficient
                val.put(term[0],1);
            } else {
                val.put(term[1],Integer.parseInt(term[0]));
            }
        }
    }
    //empty multiset
    public SMultiSet() {
        val = new HashMap<String,Integer>();
    }
    //---------------------------------------------------------------------------
    // returns a clone
    public SMultiSet clone() {
        SMultiSet copy = new SMultiSet(val.toString());
        return copy;
    }
    //---------------------------------------------------------------------------
    public String toString() {
        String val_s = "";
        
        for (HashMap.Entry<String, Integer> entry : val.entrySet()) {
            val_s = val_s + entry.getValue() + SEP + entry.getKey() + ",";
        }
        //remove trailing ","
        if (val_s.endsWith(",")) {
            val_s = val_s.substring(0, val_s.length() - 1);
        }

        return val_s;
    }
    //---------------------------------------------------------------------------
    // to add any two multisets.
    public static SMultiSet add(SMultiSet ms1,SMultiSet ms2) {
        SMultiSet res = new SMultiSet();

        for (HashMap.Entry<String, Integer> entry : ms1.val.entrySet()) {
            String key = entry.getKey();
            Integer value = entry.getValue();

            if (ms2.val.containsKey(key)) {
                Integer newV = value + ms2.val.get(key);
                if (newV != 0) {
                    res.val.put(key,newV);
                }
            }
            else {
                res.val.put(key,value);
            }
        }

        for (HashMap.Entry<String, Integer> entry : ms2.val.entrySet()) {
            String key = entry.getKey();
            Integer value = entry.getValue();

            if (!ms1.val.containsKey(key)) {
                res.val.put(key,value);
            }
        }

        return res;
    }
    // add the multiset to myself
    public void add(SMultiSet ms) {

        for (HashMap.Entry<String, Integer> entry : val.entrySet()) {
            String key = entry.getKey();
            Integer value = entry.getValue();

            if (ms.val.containsKey(key)) {
                val.put(key,value + ms.val.get(key));
            }
        }

        for (HashMap.Entry<String, Integer> entry : ms.val.entrySet()) {
            String key = entry.getKey();
            Integer value = entry.getValue();

            if (!val.containsKey(key)) {
                val.put(key,value);
            }
        }
    }

    // This is a little more efficient than the more elegant solution
    // return add(ms1, mult_k(-1,ms2))
    // Pre: ms1>=ms2
    public static SMultiSet diff(SMultiSet ms1,SMultiSet ms2) {
        SMultiSet res = new SMultiSet();

        for (HashMap.Entry<String, Integer> entry : ms1.val.entrySet()) {
            String key = entry.getKey();
            Integer value = entry.getValue();

            if (ms2.val.containsKey(key)) {
                Integer newV = value - ms2.val.get(key);
                if (newV != 0) {
                    res.val.put(key,newV);
                }
            }
            else {
                res.val.put(key,value);
            }
        }

        for (HashMap.Entry<String, Integer> entry : ms2.val.entrySet()) {
            String key = entry.getKey();
            Integer value = entry.getValue();

            if (!ms1.val.containsKey(key)) {
                res.val.put(key,-value);
            }
        }

        return res;
    }
    // Pre: val>=ms2
    // substract to myself
    public void diff(SMultiSet ms) {

        for (String key : ms.keySet()) {
            Integer value = ms.getVal(key);

            Integer newV = val.get(key) - value;
            if (newV != 0) {
                val.put(key,newV);
            }
            else {
                val.remove(key);
            }
        }
    }
    //---------------------------------------------------------------------------
    // Pre: k>=0
    public static SMultiSet mult_k(int k,SMultiSet ms) {
        SMultiSet res = new SMultiSet();

        for (HashMap.Entry<String, Integer> entry : ms.val.entrySet()) {
            String key = entry.getKey();
            Integer value = entry.getValue();

            res.val.put(key,k*value);
        }

        return res;
    }
    // Pre: k>=0
    // Multiply to myself
    public void mult_k(int k) {

        for (HashMap.Entry<String, Integer> entry : val.entrySet()) {
            String key = entry.getKey();
            Integer value = entry.getValue();

            val.put(key,k*value);
        }
    }
    //---------------------------------------------------------------------------
    // Is ms1>=ms2?
    public static boolean greaterOrEqual(SMultiSet ms1,SMultiSet ms2) {
        Set<String> keysms2 = ms2.keySet();

        for (String k : keysms2) {
            if (ms1.getVal(k) < ms2.getVal(k)) {
                return false;
            }
        }
        return true;
    }
    // Is val>=ms? (Am I ...)
    public boolean greaterOrEqual(SMultiSet ms) {
        Set<String> keysms = ms.keySet();

        for (String k : keysms) {
            if (this.getVal(k) < ms.getVal(k)) {
                return false;
            }
        }
        return true;
    }
    //---------------------------------------------------------------------------
    // Am I zero?
    public boolean isZero() {
        for (HashMap.Entry<String, Integer> entry : val.entrySet()) {
            if (0 != entry.getValue()) {
                return false;
            }
        }
        return true;
    }

    public int getVal(String key) {
        if (val.keySet().contains(key)) {
            return val.get(key);
        }
        return 0;
    }

    public void setVal(String key, int value) {
        val.put(key,value);
    }

    public void setZero() {
        val.clear();
    }

    public Set<String> keySet() {
        return val.keySet();
    }

    public boolean contains(String key) {
        return val.keySet().contains(key);
    }
    //---------------------------------------------------------------------------
    //For testing
    public static void main(String[] arg) {
        SMultiSet m1 = new SMultiSet("1'a,2'b");
        SMultiSet m2 = new SMultiSet("1'a,2'b,5'd");

        System.out.println("m1: " + m1.toString());
        System.out.println("m2: " + m2.toString());

        m1.mult_k(7);
        System.out.println("7*m1: " + m1.toString());

        SMultiSet m = new SMultiSet();
        m = m1.add(m1,m2);
        System.out.println(m.toString());

        System.out.println(m.mult_k(3,m).toString());
        System.out.println(m.diff(m1,m2).toString());

        SMultiSet zero = m.diff(m,m);
        System.out.println("zero: " + zero);

        if (zero.isZero()) {
            System.out.println("m-m is zero");
        }

        if (m.diff(m,m).isZero()) {
            System.out.println("m-m is zero again");
        }

        m.setZero();
        if (m.isZero()) {
            System.out.println("m is zero now");
        }

        m.add(m1);
        System.out.println("new m: " + m.toString());

        m1 = new SMultiSet("2'a,2'b");
        m2 = new SMultiSet("2'a,1'b");
        if (m1.greaterOrEqual(m2)) {
            System.out.println("m1 >= m2");
        }
        else {
            System.out.println("m1 < m2");
        }

        System.out.println("m1: " + m1.toString());
        System.out.println("m2: " + m2.toString());
        m1.add(m2);
        System.out.println("new m1 (m1+m2): " + m1.toString());
    }
}