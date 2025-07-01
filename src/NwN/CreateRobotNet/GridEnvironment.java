import java.util.*;
public class GridEnvironment {
    private int rows, cols;
    private SortedMap<String, Set<String>> hBag; // h(P) -> B mapping
    private SMultiSet capBag; // Capacity per place
    private SMultiSet partitionOccBag; // Occupancy per place

    public GridEnvironment(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.hBag = new TreeMap<>();
        this.capBag = new SMultiSet();
        this.partitionOccBag = new SMultiSet();

        initializeGrid();
    }

    // Initializes the grid with default values
    private void initializeGrid() {
        for (int r = 1; r <= rows; r++) {
            for (int c = 1; c <= cols; c++) {
                String place = "p" + r + "_" + c;
                capBag.setVal(place, 1); // Default: 1 robot per cell
                partitionOccBag.setVal(place, 0); // No robots at start
                hBag.put(place, new HashSet<>()); // Empty region mapping
            }
        }
    }

    public void initializeFromString(String cap, String partOcc, String h) {
        this.capBag = new SMultiSet(cap);
        this.partitionOccBag = new SMultiSet(partOcc);
        this.hBag = parseHString(h);
    }

    private SortedMap<String, Set<String>> parseHString(String hStr) {
        SortedMap<String, Set<String>> res = new TreeMap<>();
        String[] mappings = hStr.split("\\|");

        for (String entry : mappings) {
            String[] parts = entry.split(",");
            Set<String> regions = new HashSet<>();
            for (int i = 1; i < parts.length; i++) {
                regions.add(parts[i]);
            }
            res.put(parts[0], regions);
        }

        return res;
    }

    // Sets a region of interest to a cell (removes any previous region)
    public void setRegion(String place, String region) {
        if (hBag.containsKey(place)) {
            // Clear previous regions and assign the new one
            hBag.get(place).clear();
            hBag.get(place).add(region);
        } else {
            System.err.println("Invalid place: " + place);
        }
    }

    public void setRegionColumn(int col, String region) {
        for (int r = 1; r <= rows; r++) {
            String place = "p" + r + "_" + col;
            if (containsPlace(place)) {
                setRegion(place, region);
            }
        }
    }

    public void setRegionRow(int row, String region) {
        for (int c = 1; c <= cols; c++) {
            String place = "p" + row + "_" + c;
            if (containsPlace(place)) {
                setRegion(place, region);
            }
        }
    }

    public void setRegionArea(String corner1, String corner2, String region) {
        String[] parts1 = corner1.substring(1).split("_");
        String[] parts2 = corner2.substring(1).split("_");

        int row1 = Integer.parseInt(parts1[0]);
        int col1 = Integer.parseInt(parts1[1]);
        int row2 = Integer.parseInt(parts2[0]);
        int col2 = Integer.parseInt(parts2[1]);

        int startRow = Math.min(row1, row2);
        int endRow = Math.max(row1, row2);
        int startCol = Math.min(col1, col2);
        int endCol = Math.max(col1, col2);

        for (int r = startRow; r <= endRow; r++) {
            for (int c = startCol; c <= endCol; c++) {
                String place = "p" + r + "_" + c;
                if (containsPlace(place)) {
                    setRegion(place, region);
                }
            }
        }
    }

    public void setRandomRegions() {
        int totalPlaces = rows * cols;
        List<String> regions = new ArrayList<>();

        // Generate regions y1, y2, ..., yN (one per place)
        for (int i = 1; i <= totalPlaces; i++) {
            regions.add("y" + i);
        }

        Random random = new Random();

        // Assign a random region to each place (regions can repeat)
        for (int r = 1; r <= rows; r++) {
            for (int c = 1; c <= cols; c++) {
                String place = "p" + r + "_" + c;
                if (containsPlace(place)) {
                    String randomRegion = regions.get(random.nextInt(totalPlaces));
                    setRegion(place, randomRegion);
                }
            }
        }
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public void addRegion(String place, String region) {
        if (hBag.containsKey(place)) {
            hBag.get(place).add(region);  // Add the new region to the existing set
        } else {
            System.err.println("Invalid place: " + place);
        }
    }

    // Removes a specific region from a given place
    public void removeRegion(String place, String region) {
        if (hBag.containsKey(place)) {
            hBag.get(place).remove(region);
            if (hBag.get(place).isEmpty()) {
                hBag.remove(place); // Remove the place if it has no more regions
            }
        } else {
            System.err.println("Invalid place: " + place);
        }
    }


    // Sets the occupancy of a specific cell
    public void setOccupancy(String place, int robots) {
        if (partitionOccBag.contains(place)) {
            partitionOccBag.setVal(place, robots);
        } else {
            System.err.println("Invalid place: " + place);
        }
    }

    // Sets the capacity of a specific cell
    public void setCapacity(String place, int capacity) {
        if (capBag.contains(place)) {
            capBag.setVal(place, capacity);
        } else {
            System.err.println("Invalid place: " + place);
        }
    }

    public void setCapacityForRegion(String region, int capacity) {
        // Iterate through each place in the grid
        for (int r = 1; r <= rows; r++) {
            for (int c = 1; c <= cols; c++) {
                String place = "p" + r + "_" + c;

                // Check if the place has the region
                Set<String> regions = hBag.getOrDefault(place, new HashSet<>());
                if (regions.contains(region)) {
                    // Set the capacity for this place
                    setCapacity(place, capacity);
                }
            }
        }
    }

    public void setCapacityArea(String corner1, String corner2, int capacity) {
        String[] parts1 = corner1.substring(1).split("_");
        String[] parts2 = corner2.substring(1).split("_");

        int row1 = Integer.parseInt(parts1[0]);
        int col1 = Integer.parseInt(parts1[1]);
        int row2 = Integer.parseInt(parts2[0]);
        int col2 = Integer.parseInt(parts2[1]);

        int startRow = Math.min(row1, row2);
        int endRow = Math.max(row1, row2);
        int startCol = Math.min(col1, col2);
        int endCol = Math.max(col1, col2);

        for (int r = startRow; r <= endRow; r++) {
            for (int c = startCol; c <= endCol; c++) {
                String place = "p" + r + "_" + c;
                if (capBag.contains(place)) {
                    capBag.setVal(place, capacity);
                } else {
                    System.err.println("Invalid place: " + place);
                }
            }
        }
    }

    // Removes a place (P) completely from the grid
    public void removePlace(String place) {
        if (capBag.contains(place)) {
            capBag.keySet().remove(place); // Remove from capacity tracking
            partitionOccBag.keySet().remove(place); // Remove from occupancy tracking
            hBag.remove(place); // Remove from region mapping
        } else {
            System.err.println("Cannot remove: Place does not exist.");
        }
    }

    public void removeArea(String corner1, String corner2) {
        String[] parts1 = corner1.substring(1).split("_");
        String[] parts2 = corner2.substring(1).split("_");

        int row1 = Integer.parseInt(parts1[0]);
        int col1 = Integer.parseInt(parts1[1]);
        int row2 = Integer.parseInt(parts2[0]);
        int col2 = Integer.parseInt(parts2[1]);

        int startRow = Math.min(row1, row2);
        int endRow = Math.max(row1, row2);
        int startCol = Math.min(col1, col2);
        int endCol = Math.max(col1, col2);

        for (int r = startRow; r <= endRow; r++) {
            for (int c = startCol; c <= endCol; c++) {
                String place = "p" + r + "_" + c;
                if (containsPlace(place)) {
                    removePlace(place); // Removes the place from all relevant data structures
                }
            }
        }
    }

    public List<String> getValidMoves(String place) {
        List<String> validMoves = new ArrayList<>();

        // Extract row and column from place name (e.g., "p2_3" → row = 2, col = 3)
        String[] parts = place.substring(1).split("_");
        int row = Integer.parseInt(parts[0]);
        int col = Integer.parseInt(parts[1]);

        // Define neighbors (Up, Down, Left, Right)
        addIfValid(validMoves, row - 1, col); // Up
        addIfValid(validMoves, row + 1, col); // Down
        addIfValid(validMoves, row, col - 1); // Left
        addIfValid(validMoves, row, col + 1); // Right

        return validMoves;
    }

    // Helper method to add valid neighbors
    private void addIfValid(List<String> list, int r, int c) {
        String neighbor = "p" + r + "_" + c;
        if (capBag.contains(neighbor)) { // Ensure the cell exists
            list.add(neighbor);
        }
    }

    public void moveRobot(String from, String to) {
        // 1. Check if 'from' and 'to' exist in the grid
        if (!capBag.contains(from) || !capBag.contains(to)) {
            System.err.println("Invalid move: One or both places do not exist.");
            return;
        }

        // 2. Check if 'to' is a valid neighbor of 'from'
        if (!getValidMoves(from).contains(to)) {
            System.err.println("Invalid move: " + from + " cannot move to " + to);
            return;
        }

        // 3. Check if 'from' has at least one robot to move
        if (partitionOccBag.getVal(from) < 1) {
            System.err.println("Invalid move: No robots in " + from);
            return;
        }

        // 4. Check if 'to' has space (cannot exceed max capacity)
        int currentOccupancy = partitionOccBag.getVal(to);
        int maxCapacity = capBag.getVal(to);

        if (currentOccupancy + 1 > maxCapacity) {
            System.err.println("Invalid move: " + to + " is at full capacity.");
            return;
        }

        // 5. Perform the movement
        partitionOccBag.setVal(from, partitionOccBag.getVal(from) - 1);
        partitionOccBag.setVal(to, currentOccupancy + 1);

        System.out.println("Robot moved from " + from + " to " + to);
    }

    public void setGeneralCapacity(int capacity) {
        for (String place : capBag.keySet()) {
            capBag.setVal(place, capacity);
        }
    }

    public SMultiSet getPartitionOccBag() {
        return partitionOccBag;
    }

    public SMultiSet getCapacityBag() {
        return capBag;
    }

    public Set<String> getRegionSet() {
        Set<String> regions = new HashSet<>();
        for (Set<String> regionSet : hBag.values()) {
            regions.addAll(regionSet);
        }
        return regions;
    }

    public String getRegionsAsString(String place) {
        Set<String> regions = hBag.getOrDefault(place, new HashSet<>());
        return regions.isEmpty() ? "No Region" : String.join(", ", regions);
    }

    public Set<String> getRegionsForPlace(String place) {
        return hBag.getOrDefault(place, new HashSet<>());
    }

    public boolean containsPlace(String place) {
        return capBag.contains(place);
    }

    // Converts the hBag map to a formatted string inside GridEnvironment
    public String convertHBagToString() {
        StringBuilder res = new StringBuilder("\"");

        for (int r = 1; r <= rows; r++) {
            for (int c = 1; c <= cols; c++) {
                String place = "p" + r + "_" + c;
                if (hBag.containsKey(place)) {
                    res.append(place);
                    for (String region : hBag.get(place)) {
                        res.append(",").append(region);
                    }
                    res.append("|");
                }
            }
        }

        if (res.length() > 1) { // Ensure it's not empty before removing the last "|"
            res.setLength(res.length() - 1);
        }

        res.append("\"");
        return res.toString();
    }

    public String getHString() {
        return convertHBagToString();
    }


    // Returns the formatted occupancy string
    public String getOccupancyString() {
        StringBuilder res = new StringBuilder("\"");
        boolean first = true;

        for (int r = 1; r <= rows; r++) {
            for (int c = 1; c <= cols; c++) {
                String place = "p" + r + "_" + c;
                if (partitionOccBag.contains(place)) {
                    if (!first) {
                        res.append(",");
                    }
                    res.append(partitionOccBag.getVal(place)).append("'").append(place);
                    first = false;
                }
            }
        }

        res.append("\"");
        return res.toString();
    }

    public String getInitialOccupancyString() {
        StringBuilder res = new StringBuilder("\"");
        boolean first = true;

        for (int r = 1; r <= rows; r++) {
            for (int c = 1; c <= cols; c++) {
                String place = "p" + r + "_" + c;
                if (partitionOccBag.contains(place)) {
                    int occupancy = partitionOccBag.getVal(place);
                    if (occupancy > 0) {
                        if (!first) {
                            res.append(",");
                        }
                        res.append(occupancy).append("'").append(place);
                        first = false;
                    }
                }
            }
        }

        res.append("\"");
        return res.toString();
    }


    // Returns the formatted capacity string
    public String getCapacityString() {
        StringBuilder res = new StringBuilder("\"");
        boolean first = true;

        for (int r = 1; r <= rows; r++) {
            for (int c = 1; c <= cols; c++) {
                String place = "p" + r + "_" + c;
                if (capBag.contains(place)) {
                    if (!first) {
                        res.append(",");
                    }
                    res.append(capBag.getVal(place)).append("'").append(place);
                    first = false;
                }
            }
        }

        res.append("\"");
        return res.toString();
    }

    public int getCapacityValue(String place) {
        return capBag.getVal(place);  // Returns capacity or 0 if place doesn't exist
    }

    public int getOccupancyValue(String place) {
        return partitionOccBag.getVal(place);  // Returns occupancy or 0 if empty
    }

    // Displays the grid with occupancy and capacity
    public void printGrid() {
        for (int r = 1; r <= rows; r++) {
            for (int c = 1; c <= cols; c++) {
                String place = "p" + r + "_" + c;

                if (!capBag.contains(place)) { // If the place was removed
                    System.out.print("[/\\] ");
                } else {
                    int occ = partitionOccBag.getVal(place);
                    int cap = capBag.getVal(place);
                    System.out.printf("[%d/%d] ", occ, cap);
                }
            }
            System.out.println();
        }
    }


    public static void main(String[] args) {
        GridEnvironment grid = new GridEnvironment(10, 10);
        /*String cap = "1'p1_1,1'p1_2,1'p1_3,1'p1_4,1'p1_5,"
                + "1'p2_1,1'p2_2,1'p2_3,1'p2_4,1'p2_5,"
                + "1'p3_1,1'p3_2,1'p3_3,1'p3_4,1'p3_5,"
                + "1'p4_1,1'p4_2,1'p4_3,1'p4_4,1'p4_5,"
                + "1'p5_1,1'p5_2,1'p5_3,1'p5_4,1'p5_5";

        String partOcc = "1'p1_1,2'p3_3,1'p5_5"; // Some places with initial robots

        String h = "p1_1,a|p1_2,b|p1_3,c|p1_4,c|p1_5,c|"
                + "p2_1,b|p2_2,b|p2_3,w|p2_4,c|p2_5,c|"
                + "p3_1,a|p3_2,l|p3_3,m|p3_4,n|p3_5,o|"
                + "p4_1,a|p4_2,q|p4_3,r|p4_4,s|p4_5,t|"
                + "p5_1,a|p5_2,c|p5_3,w|p5_4,b|p5_5,b";

        grid.initializeFromString(cap, partOcc, h);*/

        // BROAD GRID

        /*grid.setOccupancy("p1_1", 1);

        grid.setRegionArea("p1_1", "p1_7", "y4");
        grid.setRegionArea("p1_8", "p3_10", "y3");
        grid.setRegionArea("p3_8", "p5_8", "y1");
        grid.setRegionArea("p4_9", "p6_10", "y2");
        grid.setRegionArea("p7_9", "p7_10", "y1");
        grid.setRegionArea("p8_6", "p10_10", "y7");
        grid.setRegion("p10_10", "y8");
        grid.setRegionArea("p9_7", "p9_9", "y1");
        grid.setRegion("p10_9", "y1");
        grid.setRegionArea("p6_7", "p7_8", "y6");
        grid.setRegionArea("p2_7", "p5_7", "y6");
        grid.setRegionArea("p2_6", "p5_6", "y4");
        grid.addRegion("p3_6", "y6");
        grid.setRegionArea("p6_5", "p7_6", "y1");
        grid.setRegionArea("p8_1", "p10_5", "y12");
        grid.setRegion("p10_1", "y13");
        grid.setRegionArea("p7_3", "p8_4", "y11");
        grid.setRegionArea("p7_2", "p8_2", "y1");
        grid.setRegionArea("p9_3", "p9_4", "y1");
        grid.setRegion("p7_1", "y12");
        grid.setRegionArea("p4_1", "p6_4", "y10");
        grid.setRegion("p5_5", "y10");
        grid.setRegion("p4_3", "y1");
        grid.addRegion("p5_2", "y9");
        grid.addRegion("p5_3", "y9");
        grid.setRegionArea("p2_1", "p3_1", "y4");
        grid.setRegionArea("p2_2", "p2_5", "y1");
        grid.setRegionArea("p3_5", "p4_5", "y1");
        grid.setRegionArea("p3_2", "p3_4", "y9");
        grid.addRegion("p3_2", "y4");*/


        //MAZE GRID

        /*grid.setOccupancy("p1_1", 1);

        grid.setRegionArea("p1_1", "p10_10", "y2");
        grid.setRegion("p2_1", "y1");
        grid.setRegion("p1_9", "y1");
        grid.setRegion("p5_10", "y1");
        grid.setRegion("p2_1", "y1");
        grid.setRegion("p1_3", "y1");
        grid.setRegion("p6_8", "y1");
        grid.setRegion("p8_8", "y1");
        grid.setRegionArea("p2_3", "p2_7", "y1");
        grid.setRegionArea("p3_7", "p4_7", "y1");
        grid.setRegionArea("p3_9", "p6_9", "y1");
        grid.setRegionArea("p8_9", "p9_9", "y1");
        grid.setRegionArea("p4_2", "p5_2", "y1");
        grid.setRegionArea("p9_2", "p9_6", "y1");
        grid.setRegionArea("p7_2", "p8_2", "y1");
        grid.setRegionArea("p4_4", "p7_4", "y1");
        grid.setRegion("p7_3", "y1");
        grid.setRegionArea("p4_5", "p4_6", "y1");
        grid.setRegionArea("p5_6", "p7_6", "y1");
        grid.setRegion("p1_4", "y3");
        grid.setRegion("p1_10", "y10");
        grid.setRegion("p3_6", "y4");
        grid.setRegion("p4_10", "y11");
        grid.setRegion("p6_10", "y8");
        grid.setRegion("p8_3", "y6");
        grid.setRegion("p6_3", "y7");
        grid.setRegionArea("p9_10","p10_10", "y9");
        grid.setRegion("p10_9", "y0");*/


        //HOUSE GRID

        /*grid.setOccupancy("p1_1", 1);

        grid.setRegionArea("p9_1","p10_3", "y11");
        grid.setRegionArea("p9_5","p10_6", "y10");
        grid.setRegionArea("p9_8","p10_10", "y8");
        grid.addRegion("p10_10", "y9");
        grid.setRegion("p9_10", "y1");
        grid.setRegionArea("p8_4","p10_4", "y1");
        grid.setRegionArea("p8_7","p10_7", "y1");
        grid.setRegionArea("p8_9","p8_10", "y1");
        grid.setRegion("p8_1", "y1");
        grid.setRegion("p8_3", "y1");
        grid.setRegion("p8_6", "y1");
        grid.setRegion("p8_2", "y2");
        grid.setRegion("p8_5", "y2");
        grid.setRegion("p8_8", "y2");
        grid.setRegion("p7_1", "y13");
        grid.setRegion("p7_10", "y3");
        grid.setRegionArea("p7_2","p7_9", "y2");
        grid.setRegion("p6_1", "y1");
        grid.setRegion("p4_1", "y1");
        grid.setRegion("p2_1", "y1");
        grid.setRegion("p1_1", "y13");
        grid.setRegion("p3_1", "y13");
        grid.setRegion("p5_1", "y13");
        grid.setRegion("p1_2", "y4");
        grid.setRegion("p2_3", "y1");
        grid.setRegion("p4_3", "y1");
        grid.setRegion("p6_3", "y1");
        grid.setRegion("p3_3", "y14");
        grid.setRegion("p5_3", "y14");
        grid.setRegionArea("p2_2","p6_2", "y2");
        grid.setRegionArea("p1_3","p1_5", "y9");
        grid.setRegion("p1_6", "y1");
        grid.setRegionArea("p2_4","p2_6", "y1");
        grid.setRegionArea("p3_4","p6_4", "y1");
        grid.setRegionArea("p1_8","p1_9", "y1");
        grid.setRegion("p1_7", "y5");
        grid.setRegion("p1_10", "y6");
        grid.setRegionArea("p3_5","p6_6", "y12");
        grid.addRegion("p3_6", "y9");
        grid.setRegion("p4_6", "y1");
        grid.setRegion("p6_6", "y1");
        grid.setRegionArea("p2_7","p6_10", "y7");
        grid.setRegionArea("p3_7","p6_7", "y1");
        grid.setRegion("p3_10", "y1");
        grid.setRegion("p6_10", "y1");
        grid.addRegion("p4_10", "y9");
        grid.addRegion("p5_10", "y9");*/


        //OPEN GRID

        grid.setOccupancy("p1_1", 1);

        grid.setRegion("p1_1", "y1");
        grid.setRegion("p2_2", "y1");
        grid.setRegionArea("p3_3","p4_4", "y1");
        grid.setRegionArea("p1_3","p2_3", "y2");
        grid.setRegion("p1_2", "y2");
        grid.setRegion("p2_1", "y3");
        grid.setRegionArea("p3_1","p3_2", "y3");
        grid.setRegionArea("p1_4","p2_5", "y4");
        grid.setRegionArea("p4_1","p6_2", "y7");
        grid.setRegionArea("p4_5","p5_6", "y6");
        grid.setRegionArea("p1_6","p3_6", "y5");
        grid.setRegion("p3_5", "y5");
        grid.setRegionArea("p5_3","p5_4", "y8");
        grid.setRegionArea("p6_4","p6_5", "y8");
        grid.setRegion("p7_5", "y8");
        grid.setRegionArea("p6_3","p8_3", "y9");
        grid.setRegionArea("p7_2","p7_4", "y9");
        grid.setRegionArea("p7_1","p8_1", "y10");
        grid.setRegionArea("p8_2","p9_2", "y10");
        grid.setRegionArea("p9_3","p9_4", "y10");
        grid.setRegion("p8_4", "y10");
        grid.setRegionArea("p9_1","p10_1", "y11");
        grid.setRegionArea("p10_2","p10_4", "y11");
        grid.setRegionArea("p10_5","p10_7", "y12");
        grid.setRegion("p9_6", "y12");
        grid.setRegion("p9_5", "y13");
        grid.setRegion("p9_7", "y13");
        grid.setRegionArea("p8_5","p8_7", "y13");
        grid.setRegionArea("p6_6","p7_7", "y1");
        grid.setRegionArea("p4_7","p5_7", "y1");
        grid.setRegionArea("p4_8","p10_8", "y14");
        grid.setRegionArea("p2_7","p3_8", "y17");
        grid.setRegionArea("p1_7","p2_7", "y18");
        grid.setRegionArea("p1_8","p1_9", "y18");
        grid.setRegion("p2_9", "y18");
        grid.setRegionArea("p1_10","p4_10", "y20");
        grid.setRegionArea("p3_9","p6_9", "y16");
        grid.setRegionArea("p7_9","p10_9", "y15");
        grid.setRegionArea("p5_10","p10_10", "y19");

        grid.printGrid();

        System.out.println("Regions: " + grid.getHString());
        System.out.println("Initial Occupancy: " + grid.getInitialOccupancyString());
        System.out.println("Capacity: " + grid.getCapacityString());


    }
}
