import java.io.*;
import java.util.*;

public class GridToPetriNet {
    private GridEnvironment grid;
    private Map<String, Set<String>> mergedPlaces; // Merged places -> Original grid cells
    private Map<String, Set<String>> connectivityMap; // Merged places -> Connected merged places
    private Map<String, String> mergedPlaceRegions; // Merged places -> Region
    private Map<String, Integer> mergedPlaceOccupancy; // Merged places -> Total occupancy
    private Map<String, Integer> mergedPlaceCapacity; // Merged places -> Average capacity

    public GridToPetriNet(GridEnvironment grid) {
        this.grid = grid;
        this.mergedPlaces = new HashMap<>();
        this.connectivityMap = new HashMap<>();
        this.mergedPlaceRegions = new HashMap<>();
        this.mergedPlaceOccupancy = new HashMap<>();
        this.mergedPlaceCapacity = new HashMap<>();
        mergeRegions();
        createConnectivityMap();
    }

    // Perform a flood-fill (BFS) to merge regions into unique places
    private void mergeRegions() {
        Set<String> visited = new HashSet<>();
        int placeIndex = 1; // To assign unique place names (p1, p2, ...)

        for (int r = 1; r <= grid.getRows(); r++) {
            for (int c = 1; c <= grid.getCols(); c++) {
                String startCell = "p" + r + "_" + c;
                if (!visited.contains(startCell) && grid.containsPlace(startCell)) {
                    String region = grid.getRegionsAsString(startCell);
                    if (!region.equals("No Region")) {
                        String newPlace = "p" + placeIndex++;
                        Set<String> groupedCells = floodFill(startCell, region, visited);
                        mergedPlaces.put(newPlace, groupedCells);
                        mergedPlaceRegions.put(newPlace, region); // Store the region

                        // Compute occupancy and capacity
                        int totalOccupancy = 0;
                        double totalCapacity = 0;
                        for (String cell : groupedCells) {
                            totalOccupancy += grid.getOccupancyValue(cell);
                            totalCapacity += grid.getCapacityValue(cell);
                        }
                        mergedPlaceOccupancy.put(newPlace, totalOccupancy);
                        mergedPlaceCapacity.put(newPlace, (int) (totalCapacity / groupedCells.size())); // Truncate decimal

                    }
                }
            }
        }
    }

    // BFS-based flood fill to find all connected cells of the same region
    private Set<String> floodFill(String start, String region, Set<String> visited) {
        Set<String> groupedCells = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        queue.add(start);
        visited.add(start);

        while (!queue.isEmpty()) {
            String cell = queue.poll();
            groupedCells.add(cell);

            for (String neighbor : grid.getValidMoves(cell)) {
                if (!visited.contains(neighbor) && grid.containsPlace(neighbor)
                        && grid.getRegionsAsString(neighbor).equals(region)) {
                    queue.add(neighbor);
                    visited.add(neighbor);
                }
            }
        }

        return groupedCells;
    }

    // Create a connectivity map between merged places
    private void createConnectivityMap() {
        for (String place1 : mergedPlaces.keySet()) {
            connectivityMap.put(place1, new HashSet<>());

            for (String cell : mergedPlaces.get(place1)) {
                for (String neighbor : grid.getValidMoves(cell)) {
                    String neighborRegion = grid.getRegionsAsString(neighbor);
                    if (!neighborRegion.equals("No Region")) {
                        String mergedNeighbor = findMergedPlaceForCell(neighbor);
                        if (mergedNeighbor != null && !mergedNeighbor.equals(place1)) {
                            connectivityMap.get(place1).add(mergedNeighbor);
                        }
                    }
                }
            }
        }
    }

    // Find the merged place corresponding to a given original grid cell
    private String findMergedPlaceForCell(String cell) {
        for (Map.Entry<String, Set<String>> entry : mergedPlaces.entrySet()) {
            if (entry.getValue().contains(cell)) {
                return entry.getKey();
            }
        }
        return null;
    }

    // Display the merged places and their original grid cells
    public void printMergedPlaces() {
        System.out.println("Merged Places:");
        for (Map.Entry<String, Set<String>> entry : mergedPlaces.entrySet()) {
            System.out.println(entry.getKey() + " -> " + entry.getValue());
        }
    }

    // Display the regions associated with each merged place
    public void printMergedRegions() {
        System.out.println("\nMerged Place Regions:");
        for (Map.Entry<String, String> entry : mergedPlaceRegions.entrySet()) {
            System.out.println(entry.getKey() + " belongs to region: " + entry.getValue());
        }
    }

    // Display occupancy and capacity of each merged place
    public void printOccupancyAndCapacity() {
        System.out.println("\nOccupancy & Capacity:");
        for (String place : mergedPlaces.keySet()) {
            System.out.println(place + " -> Occupancy: " + mergedPlaceOccupancy.get(place) +
                    ", Capacity: " + mergedPlaceCapacity.get(place));
        }
    }

    // Display the connectivity between merged places
    public void printConnectivityMap() {
        System.out.println("\nConnectivity Map:");
        for (Map.Entry<String, Set<String>> entry : connectivityMap.entrySet()) {
            System.out.println(entry.getKey() + " connected to " + entry.getValue());
        }
    }

    // Getters for accessing the merged places, connectivity, regions, occupancy, and capacity
    public Map<String, Set<String>> getMergedPlaces() {
        return mergedPlaces;
    }

    public Map<String, Set<String>> getConnectivityMap() {
        return connectivityMap;
    }

    public Map<String, String> getMergedPlaceRegions() {
        return mergedPlaceRegions;
    }

    public Map<String, Integer> getMergedPlaceOccupancy() {
        return mergedPlaceOccupancy;
    }

    public Map<String, Integer> getMergedPlaceCapacity() {
        return mergedPlaceCapacity;
    }

    public String generatePNML() {
        StringBuilder pnml = new StringBuilder();

        pnml.append("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n");
        pnml.append("<pnml xmlns=\"http://www.pnml.org/version-2009/grammar/pnml\">\n");
        pnml.append("  <net id=\"grid_net\" type=\"RefNet\">\n");


        for (String place : mergedPlaces.keySet()) {
            pnml.append("      <place id=\"" + place + "\">\n");

            // Only add initialMarking if occupancy is greater than 0
            if (mergedPlaceOccupancy.getOrDefault(place, 0) > 0) {
                pnml.append(" <initialMarking>\n");
                //pnml.append(" <text>[" + mergedPlaceOccupancy.get(place) + "]</text>\n");
                pnml.append(" <text>[]</text>\n");
                pnml.append(" </initialMarking>\n");
            }

            // Add the name and graphics elements
            pnml.append("        <name>\n");
            pnml.append("          <graphics>\n");
            pnml.append("            <offset x=\"35\" y=\"-35\"/>\n");
            pnml.append("          </graphics>\n");
            pnml.append("          <text>" + place + " (" + mergedPlaceRegions.get(place) + ")</text>\n");
            pnml.append("        </name>\n");

            pnml.append("      </place>\n");
        }

        // Start transition numbering from the next number after the last place (p1 to p5 -> t6, t7, t8...)
        int transitionCounter = mergedPlaces.size() + 1;
        int arcCounter = 0; // To generate unique arc IDs
        Random random = new Random();

        for (String place1 : connectivityMap.keySet()) {
            for (String place2 : connectivityMap.get(place1)) {
                String fromRegion = mergedPlaceRegions.get(place1);
                String toRegion = mergedPlaceRegions.get(place2);
                String transitionId = "t" + transitionCounter++; // Increment the transition ID

                pnml.append("      <transition id=\"" + transitionId + "\">\n");

                // Transition graphics
                pnml.append("        <graphics>\n");
                pnml.append("          <position x=\"0\" y=\"0\"/>\n");
                pnml.append("          <dimension x=\"20\" y=\"20\"/>\n");
                pnml.append("          <fill color=\"rgb(112,219,147)\"/>\n");
                pnml.append("          <line color=\"rgb(0,0,0)\"/>\n");
                pnml.append("        </graphics>\n");

                // Uplink section
                pnml.append("        <uplink>\n");
                pnml.append("          <graphics>\n");
                pnml.append("            <offset x=\"0\" y=\"-18\"/>\n");
                pnml.append("          </graphics>\n");
                // Transition condition with random "time" value between 1 and 10
                int time = random.nextInt(10) + 1; // Random value between 1 and 10
                //pnml.append("          <text>:r(\"" + toRegion + "\",\"1'" + place2 + "\",\"1'" + place1 + "\",\"" + time + "\")</text>\n");
                pnml.append("          <text>:r(\"" + toRegion + "\",\"" + fromRegion + "\",\"1'" + place2 + "\",\"1'" + place1 + "\",\"" + time + "\")</text>\n");

                pnml.append("        </uplink>\n");
                pnml.append("      </transition>\n");

                // Arcs from place1 to transition
                pnml.append("      <arc id=\"a" + arcCounter++ + "\" source=\"" + place1 + "\" target=\"" + transitionId + "\"/>\n");

                // Arcs from transition to place2
                pnml.append("      <arc id=\"a" + arcCounter++ + "\" source=\"" + transitionId + "\" target=\"" + place2 + "\"/>\n");
            }
        }

        pnml.append("  </net>\n");
        // End the PNML document
        pnml.append("</pnml>");

        return pnml.toString();
    }

    // Method to save the PNML output to a .pnml file
    public void saveToFile(String filename) throws IOException {
        String pnml = generatePNML();

        // Create a FileWriter to save the PNML content to a file
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write(pnml);
        } catch (IOException e) {
            throw new IOException("Error writing PNML to file: " + e.getMessage());
        }
    }

    // Method to generate the Regions string
    public String generateRegions() {
        StringBuilder regionString = new StringBuilder();

        // Sort places like p1, p2, ..., p10, etc.
        List<String> sortedPlaces = new ArrayList<>(mergedPlaceRegions.keySet());
        sortedPlaces.sort(Comparator.comparingInt(p -> Integer.parseInt(p.substring(1))));

        for (String place : sortedPlaces) {
            String region = mergedPlaceRegions.get(place).replace(" ", ""); // 🔧 remove spaces
            regionString.append(place).append(",").append(region).append("|");
        }

        // Remove trailing "|"
        if (regionString.length() > 0) {
            regionString.deleteCharAt(regionString.length() - 1);
        }

        return "\"" + regionString.toString() + "\"";
    }

    // Method to generate the Initial Occupancy string (only for places with occupancy > 0)
    public String generateInitialOccupancy() {
        StringBuilder occupancyString = new StringBuilder();

        for (Map.Entry<String, Integer> entry : mergedPlaceOccupancy.entrySet()) {
            String place = entry.getKey();
            int occupancy = entry.getValue();

            // Include the place and its actual occupancy value if the occupancy > 0
            if (occupancy > 0) {
                // Format it as <occupancy_value>'<place_name>
                occupancyString.append(occupancy).append("'").append(place).append(",");
            }
        }

        // Remove the trailing "," if it exists
        if (occupancyString.length() > 0) {
            occupancyString.deleteCharAt(occupancyString.length() - 1);
        }

        return "\"" + occupancyString.toString() + "\"";
    }

    public String generateCapacity() {
        StringBuilder capacity = new StringBuilder();

        // Sort the place keys in natural order
        List<String> sortedPlaces = new ArrayList<>(mergedPlaces.keySet());
        sortedPlaces.sort(Comparator.comparingInt(p -> Integer.parseInt(p.substring(1))));

        for (int i = 0; i < sortedPlaces.size(); i++) {
            String place = sortedPlaces.get(i);
            int placeCapacity = mergedPlaceCapacity.get(place);
            capacity.append(placeCapacity).append("'").append(place);

            if (i < sortedPlaces.size() - 1) {
                capacity.append(",");
            }
        }

        return "\"" + capacity.toString() + "\"";
    }

    // MAIN METHOD TO TEST FUNCTIONALITY
    public static void main(String[] args) {
        // Example: Creating a simple grid
        //GridEnvironment grid = new GridEnvironment(10, 11);

        //HOUSE GRID

        /*grid.setOccupancy("p10_6",2);

        grid.setRegionArea("p1_1","p1_11", "y1");
        grid.setRegionArea("p1_1","p11_1", "y1");
        grid.setRegionArea("p11_11","p1_11", "y1");
        grid.setRegionArea("p11_11","p11_1", "y1");
        grid.setRegionArea("p1_3","p2_3", "y1");
        grid.setRegionArea("p1_5","p2_5", "y1");
        grid.setRegionArea("p1_8","p6_8", "y1");
        grid.setRegionArea("p9_8","p11_8", "y1");
        grid.setRegionArea("p9_5","p11_5", "y1");
        grid.setRegionArea("p3_10","p3_11", "y1");
        grid.setRegionArea("p5_10","p5_11", "y1");
        grid.setRegionArea("p5_1","p5_2", "y1");
        grid.setRegionArea("p4_4","p7_4", "y1");
        grid.setRegionArea("p4_4","p4_5", "y1");
        grid.setRegionArea("p6_4","p6_6", "y1");
        grid.setRegionArea("p6_6","p7_6", "y1");

        grid.setRegionArea("p8_2","p10_4", "y8");
        grid.setRegionArea("p8_5","p7_5","y8");
        grid.addRegion("p7_5","y9");

        grid.setRegionArea("p6_2","p7_3", "y10");
        grid.setRegionArea("p5_3","p7_3", "y10");

        grid.setRegionArea("p3_2","p3_5", "y11");
        grid.setRegionArea("p4_2","p4_3", "y11");
        grid.setRegion("p2_2","y11");
        grid.setRegion("p2_4","y11");
        grid.addRegion("p2_2","y12");
        grid.addRegion("p2_4","y13");

        grid.setRegionArea("p2_6","p5_7", "y6");
        grid.setRegion("p5_5","y6");
        grid.addRegion("p5_5","y7");

        grid.setRegionArea("p10_6","p8_7","y5");
        grid.setRegionArea("p7_7","p8_8","y5");
        grid.setRegion("p6_7","y5");

        grid.setRegionArea("p6_9","p10_10","y2");
        grid.setRegionArea("p2_9","p5_9","y2");
        grid.setRegion("p2_10","y2");
        grid.setRegion("p4_10","y2");

        grid.addRegion("p5_9","y3");
        grid.addRegion("p4_9","y3");
        grid.addRegion("p4_10","y3");

        grid.addRegion("p2_9","y4");
        grid.addRegion("p2_10","y4");
        grid.addRegion("p3_9","y4");

        grid.setCapacityForRegion("y5",2);
        grid.setCapacityForRegion("y6",2);*/

        //-----------------------------------------------------------------------------------------

        //WAREHOUSE
        /*grid.setOccupancy("p5_5",4);


        grid.setRegionArea("p1_1","p5_1", "y1");
        grid.setRegionArea("p1_3","p5_3", "y1");
        grid.setRegionArea("p1_5","p5_5", "y1");
        grid.setRegionArea("p1_7","p5_7", "y1");
        grid.setRegionArea("p1_9","p5_9", "y1");
        grid.setRegionArea("p1_11","p5_11", "y1");
        grid.setRegionArea("p4_1","p4_11", "y1");

        grid.setRegionArea("p6_1","p10_1", "y2");
        grid.setRegionArea("p6_3","p10_3", "y2");
        grid.setRegionArea("p6_5","p10_5", "y2");
        grid.setRegionArea("p6_7","p10_7", "y2");
        grid.setRegionArea("p6_9","p10_9", "y2");
        grid.setRegionArea("p6_11","p10_11", "y2");
        grid.setRegionArea("p7_1","p7_11", h"y2");

        grid.setRegionArea("p1_2","p3_2", "y3");

        grid.setRegionArea("p1_4","p3_4", "y4");

        grid.setRegionArea("p1_6","p3_6", "y5");

        grid.setRegionArea("p1_8","p3_8", "y6");

        grid.setRegionArea("p1_10","p3_10", "y7");

        grid.setRegionArea("p5_2","p6_2", "y8");

        grid.setRegionArea("p5_4","p6_4", "y9");

        grid.setRegionArea("p5_6","p6_6", "y10");

        grid.setRegionArea("p5_8","p6_8", "y11");

        grid.setRegionArea("p5_10","p6_10", "y12");

        grid.setRegionArea("p8_2","p10_2", "y13");

        grid.setRegionArea("p8_4","p10_4", "y14");

        grid.setRegionArea("p8_6","p10_6", "y15");

        grid.setRegionArea("p8_8","p10_8", "y16");

        grid.setRegionArea("p8_10","p10_10", "y17");

        grid.setCapacityForRegion("y1",4);
        grid.setCapacityForRegion("y2",4);*/

        //---------------------------------------------------------------------------------------------

        //FREE SPACE

        /*grid.setOccupancy("p1_1",2);

        grid.setRegionArea("p1_1","p3_3", "y1");

        grid.setRegionArea("p1_4","p3_6", "y2");

        grid.setRegionArea("p2_7","p3_8", "y3");

        grid.setRegionArea("p1_7","p1_10", "y4");
        grid.setRegionArea("p2_9","p6_10", "y4");

        grid.setRegionArea("p4_6","p7_8", "y5");

        grid.setRegionArea("p4_4","p5_5", "y6");

        grid.setRegionArea("p6_4","p7_5", "y7");

        grid.setRegionArea("p4_1","p6_3", "y8");

        grid.setRegionArea("p7_1","p9_1", "y9");

        grid.setRegionArea("p7_2","p9_3", "y10");

        grid.setRegionArea("p8_5","p9_10", "y11");
        grid.setRegionArea("p7_9","p7_10", "y11");

        grid.setCapacityForRegion("y1",2);*/

        //----------------------------------------------------------------------------------------------

        //CITY

        /*GridEnvironment grid = new GridEnvironment(10, 11);

        grid.removeArea("p2_2","p3_2");
        grid.removeArea("p2_4","p3_4");
        grid.removeArea("p2_6","p3_6");
        grid.removeArea("p2_8","p3_8");
        grid.removeArea("p2_10","p3_10");

        grid.removeArea("p5_2","p6_2");
        grid.removeArea("p5_4","p6_6");
        grid.removeArea("p5_8","p6_8");
        grid.removeArea("p5_10","p6_10");

        grid.removeArea("p8_2","p9_2");
        grid.removeArea("p8_4","p9_4");
        grid.removeArea("p8_6","p9_6");
        grid.removeArea("p8_8","p9_10");

        grid.setRegionArea("p1_1","p1_11","y1");
        grid.setRegionArea("p4_1","p4_11","y2");
        grid.setRegionArea("p7_1","p7_11","y3");
        grid.setRegionArea("p10_1","p10_11","y4");

        grid.setRegionArea("p2_1","p3_1","y5");
        grid.setRegionArea("p5_1","p6_1","y5");
        grid.setRegionArea("p8_1","p9_1","y5");
        grid.setRegion("p1_1","y11");
        grid.setRegion("p4_1","y17");
        grid.setRegion("p7_1","y23");
        grid.setRegion("p10_1","y29");

        grid.setRegionArea("p2_3","p3_3","y6");
        grid.setRegionArea("p5_3","p6_3","y6");
        grid.setRegionArea("p8_3","p9_3","y6");
        grid.setRegion("p1_3","y12");
        grid.setRegion("p4_3","y18");
        grid.setRegion("p7_3","y24");
        grid.setRegion("p10_3","y30");

        grid.setRegionArea("p2_5","p3_5","y7");
        grid.setRegionArea("p8_5","p9_5","y7");
        grid.setRegion("p1_5","y13");
        grid.setRegion("p4_5","y19");
        grid.setRegion("p7_5","y25");
        grid.setRegion("p10_5","y31");

        grid.setRegionArea("p2_7","p3_7","y8");
        grid.setRegionArea("p5_7","p6_7","y8");
        grid.setRegionArea("p8_7","p9_7","y8");
        grid.setRegion("p1_7","y14");
        grid.setRegion("p4_7","y20");
        grid.setRegion("p7_7","y26");
        grid.setRegion("p10_7","y32");

        grid.setRegionArea("p2_9","p3_9","y9");
        grid.setRegionArea("p5_9","p6_9","y9");
        grid.setRegion("p1_9","y15");
        grid.setRegion("p4_9","y21");
        grid.setRegion("p7_9","y27");
        grid.setRegion("p10_9","y33");

        grid.setRegionArea("p2_11","p3_11","y10");
        grid.setRegionArea("p5_11","p6_11","y10");
        grid.setRegionArea("p8_11","p9_11","y10");
        grid.setRegion("p1_11","y16");
        grid.setRegion("p4_11","y22");
        grid.setRegion("p7_11","y28");
        grid.setRegion("p10_11","y34");

        grid.setCapacityForRegion("y1",4);
        grid.setCapacityForRegion("y2",4);
        grid.setCapacityForRegion("y3",4);
        grid.setCapacityForRegion("y4",4);

        grid.setCapacityForRegion("y5",2);
        grid.setCapacityForRegion("y6",2);
        grid.setCapacityForRegion("y7",2);
        grid.setCapacityForRegion("y8",2);
        grid.setCapacityForRegion("y9",2);
        grid.setCapacityForRegion("y10",2);

        grid.setOccupancy("p1_6",4);*/

        //----------------------------------------------------------------------------------------------

        //SHOPPING MALL

        /*GridEnvironment grid = new GridEnvironment(10, 13);

        grid.removeArea("p1_6","p3_6");
        grid.removeArea("p3_4","p3_6");
        grid.removeArea("p3_4","p5_4");
        grid.removeArea("p5_4","p5_6");
        grid.removeArea("p4_1","p4_2");
        grid.removePlace("p4_8");
        grid.removePlace("p3_11");

        grid.removeArea("p8_2","p10_2");
        grid.removeArea("p8_4","p10_4");
        grid.removeArea("p8_6","p10_6");

        grid.removeArea("p1_10","p3_10");
        grid.removeArea("p4_9","p6_9");
        grid.removePlace("p4_13");
        grid.removePlace("p7_13");

        grid.setRegionArea("p1_3","p7_3","y1");
        grid.setRegionArea("p7_3","p7_12","y1");
        grid.setRegionArea("p1_12","p10_12","y1");
        grid.setRegionArea("p7_8","p10_8","y1");
        grid.setRegionArea("p1_7","p7_7","y1");
        grid.setRegion("p4_6","y1");

        grid.setCapacityForRegion("y1",4);

        grid.setRegionArea("p1_1","p3_2","y2");
        grid.setRegionArea("p5_1","p7_2","y3");
        grid.setRegionArea("p8_1","p10_1","y18");
        grid.setRegionArea("p8_3","p10_3","y4");
        grid.setRegionArea("p8_5","p10_5","y5");
        grid.setRegionArea("p8_7","p10_7","y6");
        grid.setRegionArea("p8_9","p10_11","y7");
        grid.setRegionArea("p8_13","p10_13","y8");
        grid.setRegionArea("p5_13","p6_13","y9");
        grid.setRegionArea("p1_13","p3_13","y10");
        grid.setRegionArea("p6_4","p6_6","y11");
        grid.setRegion("p4_5","y12");
        grid.setRegionArea("p1_4","p2_5","y13");
        grid.setRegionArea("p1_8","p3_9","y14");
        grid.setRegionArea("p5_8","p6_8","y15");
        grid.setRegionArea("p4_10","p6_11","y16");
        grid.setRegionArea("p1_11","p2_11","y17");

        grid.setOccupancy("p1_7",4);*/

        //----------------------------------------------------------------------------------------------

        //HOME

        /*GridEnvironment grid = new GridEnvironment(9, 6);

        grid.removeArea("p1_5","p6_6");
        grid.removeArea("p5_3","p6_4");
        grid.removeArea("p9_2","p9_6");
        grid.removePlace("p2_4");
        grid.removePlace("p4_4");
        grid.removePlace("p4_2");
        grid.removePlace("p6_2");
        grid.removePlace("p7_3");
        grid.removePlace("p7_5");

        grid.setRegionArea("p1_1","p3_3","y8");
        grid.setRegionArea("p4_1","p8_1","y2");
        grid.setRegionArea("p8_2","p8_6","y2");
        grid.setRegion("p9_1","y1");
        grid.setRegion("p7_6","y3");
        grid.setRegion("p7_4","y4");
        grid.setRegion("p7_2","y7");
        grid.setRegion("p5_2","y5");
        grid.setRegion("p4_3","y6");
        grid.setRegion("p3_4","y10");
        grid.setRegion("p1_4","y9");

        grid.setCapacityForRegion("y1",4);
        grid.setCapacityForRegion("y2",4);
        grid.setCapacityForRegion("y8",4);

        grid.setOccupancy("p9_1",4);*/

        //----------------------------------------------------------------------------------------------

        //OFFICE

        GridEnvironment grid = new GridEnvironment(5, 8);

        grid.removeArea("p2_6","p2_7");
        grid.removeArea("p4_6","p4_7");
        grid.removePlace("p2_2");
        grid.removePlace("p4_2");

        grid.setRegion("p3_8","y1");
        grid.setRegion("p3_1","y8");
        grid.setRegion("p3_2","y3");
        grid.setRegionArea("p2_3","p4_4","y3");
        grid.setRegionArea("p3_5","p3_7","y2");
        grid.setRegionArea("p1_1","p2_1","y6");
        grid.setRegionArea("p1_1","p1_2","y6");
        grid.setRegionArea("p5_1","p4_1","y7");
        grid.setRegionArea("p5_1","p5_2","y7");
        grid.setRegionArea("p1_5","p1_8","y4");
        grid.setRegionArea("p1_8","p2_8","y4");
        grid.setRegionArea("p5_5","p5_8","y5");
        grid.setRegionArea("p5_8","p4_8","y5");
        grid.setRegionArea("p1_3","p1_4","y9");
        grid.setRegionArea("p5_3","p5_4","y10");

        grid.setCapacityForRegion("y1",4);
        grid.setCapacityForRegion("y2",4);
        grid.setCapacityForRegion("y3",4);
        grid.setCapacityForRegion("y4",4);
        grid.setCapacityForRegion("y5",4);
        grid.setCapacityForRegion("y6",4);
        grid.setCapacityForRegion("y7",4);
        grid.setCapacityForRegion("y8",4);
        grid.setCapacityForRegion("y9",4);
        grid.setCapacityForRegion("y10",4);

        grid.setOccupancy("p3_8",4);

        //----------------------------------------------------------------------------------------------

        // Create a GridToPetriNet instance
        GridToPetriNet petriNet = new GridToPetriNet(grid);
        // Generate PNML
        String pnml = petriNet.generatePNML();

        try {
            petriNet.saveToFile("output.pnml");
            System.out.println("PNML saved to output.pnml");
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }
        // Print the merged places and connectivity map
        petriNet.printMergedPlaces();
        petriNet.printMergedRegions();
        petriNet.printConnectivityMap();
        petriNet.printOccupancyAndCapacity();

        System.out.println("Regions: " + petriNet.generateRegions());
        System.out.println("Initial Occupancy: " + petriNet.generateInitialOccupancy());
        System.out.println("Capacity: " + petriNet.generateCapacity());
    }
}
