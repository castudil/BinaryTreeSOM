/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package binarytreesom.clustering;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import weka.core.Instances;

/**
 *
 * @author castudillo
 */
public class BinaryTreeSOMClusteringTest {
    

            
    public BinaryTreeSOMClusteringTest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
    }
    
    @AfterClass
    public static void tearDownClass() {
    }
    
    @Before
    public void setUp() {
      
    }
    
    @After
    public void tearDown() {
    }

    /**
     * Test of initialize method, of class BinaryTreeSOMClustering.
     */
    @Test
    public void testInitialize() throws Exception {
        System.out.println("initialize");
        BinaryTreeSOMClustering instance;
        //System.out.println("Working Directory = " +System.getProperty("user.dir"));
        instance = new BinaryTreeSOMClustering("data/iris.arff", 1000, 0.2, 0.0, 3, 1);
        //instance.initialize();
        assertFalse(instance.getFilenameARFF().equals("asdasdasdasd"));
        assertTrue(instance.getFilenameARFF().equals("data/iris.arff"));
        assertEquals(instance.getMaximumNumberofIterations(),1000);
        assertEquals(instance.getHeightOfTheTree(),3);
        assertEquals(instance.getCurrentAlpha(),0.2,0.0001);
        assertEquals(instance.getInitialAlpha(),0.2,0.0001);
        assertEquals(instance.getFinalAlpha(),0.0,0.0001);
        assertEquals(instance.getNumberOfNeurons(),7);
        assertEquals(instance.getNumberOfInstances(),150);
        assertNotNull(instance.getWeight());
        double [][] weight=instance.getWeight();
        assertEquals(instance.getNumberOfNeurons(),weight.length);
        assertEquals(instance.getDimensionality(),weight[0].length);
        //System.out.println(Arrays.toString(weight[0]));
        System.out.println("weights: "+Arrays.deepToString(weight));
    }

    /**
     * Test of update method, of class BinaryTreeSOMClustering.
     */
    @Test
    public void testUpdate() throws IOException {
        System.out.println("update");
        //double alpha = 0.0;
        
        final double[] x = {0.0,0.0};
        final double[] w = {1.0,1.0};
        
        
        BinaryTreeSOMClustering instance = new BinaryTreeSOMClustering("data/iris.arff", 0, 0.0, 0.0, 0, 0);
        
        final double [] result1={1.0,1.0};
        final double epsilon=0.001;
        assertArrayEquals(result1,w,epsilon);
        instance.update(0.0, x, w);
        final double [] result2={1.0,1.0};
        assertArrayEquals(result2,w,epsilon);
        instance.update(1.0, x, w);
        final double [] result3={0.0,0.0};
        assertArrayEquals(result3,w,epsilon);
        
        final double[] x4 = {0.0,0.0};
        final double[] w4 = {100.0,50.0};
        instance.update(0.5, x4, w4);
        final double [] result4={50.0,25.0};
        assertArrayEquals(result4,w4,epsilon);
     
        //test Nan

        final double[] x5 = {2.0,Double.NaN};
        final double[] w5 = {1.0,1.0};
        BinaryTreeSOMClustering.update(0.5, x5, w5);
        final double [] result5={1.5,1.0};
        assertArrayEquals(result5,w5,epsilon);

        
    }

    /**
     * Test of findBest method, of class BinaryTreeSOMClustering.
     */
    @Test
    public void testFindBest() throws IOException {
        System.out.println("findBest");
        BinaryTreeSOMClustering instance = new BinaryTreeSOMClustering("data/iris.arff", 0, 0.0, 0.0, 2, 0);
        double [][]w=instance.getWeight();
        //System.out.println("weights before findbest: "+Arrays.deepToString(w));
        
        double[] x1 = w[2];
        int index1=instance.findBest(x1);
        assertEquals(2, index1);
        
        double[] x2 = w[1];
        int index2=instance.findBest(x2);
        assertEquals(1, index2);
        
        
        double[] x3 = {0.0,0.0,0.0,0.0,0.0};
        instance = new BinaryTreeSOMClustering("data/iris.arff", 1000, 0.5, 0.5, 4, 1);
        System.out.println("weights before findbest: "+Arrays.deepToString(instance.getWeight()));
        assertEquals(instance.getWeight().length,15);
        int index3=instance.findBest(x3);
        System.out.println("weighst  after findbest: "+Arrays.deepToString(instance.getWeight()));
        assertEquals(8, index3);
    }

    /**
     * Test of readArff method, of class BinaryTreeSOMClustering.
     * @throws java.lang.Exception
     */
    @Test
    public void testReadArff() throws Exception {
        System.out.println("readArff");
        String filename = "data/iris.arff";
        String expResult = "iris";
        Instances result = BinaryTreeSOMClustering.readArff(filename);
        assertNotNull(result);
        assertEquals(expResult, result.relationName());
        assertEquals(150, result.numInstances());
        assertEquals(5,result.numAttributes());//including class
    }

    /**
     * Test of learn method, of class BinaryTreeSOMClustering.
     * @throws java.io.IOException
     */
    @Test
    public void testLearn() throws IOException {
        System.out.println("learn");
        BinaryTreeSOMClustering instance;
        instance = new BinaryTreeSOMClustering("data/iris.arff", 10000000, 0.5, 0.0, 4, 1);

        instance.learn();
        
        //root should be aproximately the mean
        //scompute the mean
        
        double[][] w = instance.getWeight();
        double mean[]=new double[w[0].length];
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                mean[j]+=w[i][j];
            }
        }
        for (int j = 0; j < w[0].length; j++)
            mean[j]/=w.length;
        
        System.out.println("   mean before findbest: "+Arrays.toString(mean));
        System.out.println("weights before findbest: "+Arrays.toString(w[0]));
        assertArrayEquals(mean,w[0],0.1);
    }

    /**
     * Test of computeCurrentParameterValue method, of class BinaryTreeSOMClustering.
     */
    @Test
    public void testComputeCurrentParameterValue() throws IOException {
        System.out.println("computeCurrentParameterValue");
        BinaryTreeSOMClustering instance = new BinaryTreeSOMClustering("data/iris.arff", 0, 0.0, 0.0, 0, 0);

        double currentValue = 0.0;
        double initialValue = 0.0;
        double finalValue = 100;
        int maximumNumberofIterations = 10;
        double expResult = 10.0;
        double result = instance.computeCurrentParameterValue(currentValue, initialValue, finalValue, maximumNumberofIterations);
        assertEquals(expResult, result, 0.0);

        currentValue=10.0;
        expResult = 20.0;
        result = instance.computeCurrentParameterValue(currentValue, initialValue, finalValue, maximumNumberofIterations);
        assertEquals(expResult, result, 0.0);
        
        currentValue=20.0;
        expResult = 30.0;
        result = instance.computeCurrentParameterValue(currentValue, initialValue, finalValue, maximumNumberofIterations);
        assertEquals(expResult, result, 0.0);
        
        currentValue = 1.0;
        initialValue = 1.0;
        finalValue = 0.0;
        maximumNumberofIterations = 100;
        result=BinaryTreeSOMClustering.computeCurrentParameterValue(currentValue, initialValue, finalValue, maximumNumberofIterations);
        expResult=0.99;
        assertEquals(expResult, result,0.0001);

        currentValue = 0.2;
        initialValue = 1.0;
        finalValue = 0.0;
        maximumNumberofIterations = 10;
        result=BinaryTreeSOMClustering.computeCurrentParameterValue(currentValue, initialValue, finalValue, maximumNumberofIterations);
        expResult=0.1;
        assertEquals(expResult, result,0.0001);
    
    }
 
    @Test
    public void testweights() throws IOException {
        BinaryTreeSOMClustering instance = new BinaryTreeSOMClustering("data/iris.arff", 500000, 0.5, 0.0, 4, 1);
        instance.learn();
        double[][] w = instance.getWeight();
        BufferedWriter br = new BufferedWriter(new FileWriter("data/weights.csv"));
        StringBuilder sb = new StringBuilder();
        //for (String element : w[0]) {
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                sb.append(""+w[i][j]);
                sb.append(",");
            }
            sb.append("\n\r");
        }
        br.write(sb.toString());
        br.close();
    }
    
}
