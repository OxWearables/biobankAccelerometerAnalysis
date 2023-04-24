import org.junit.Test;

import java.time.Instant;
import java.time.ZonedDateTime;
import java.time.ZoneId;
import java.time.Instant;

import static org.junit.Assert.assertEquals;

public class ActigraphTest {

    /*
    The ground truth of the .Net time tick conversion was generated using
    https://github.com/THLfi/read.gt3x/blob/c4b83cc262c7b3dc66b0408f9c72123686dd3e33/R/utils.R#L128-L133
     */
    @Test
    public void convertNetTime2LocalZonedDatetimeStandard() {
        long myTime = 635887368000000000L;
        String localTimeZone = "UTC";
        long localTimeMillis = ActigraphReader.GT3XfromTickToMillisecond(myTime);
        Instant computedTimeIns = Instant.ofEpochMilli(localTimeMillis);
        ZonedDateTime computedTime = computedTimeIns.atZone(ZoneId.of("UTC"));
        ZonedDateTime targetTime =
                ZonedDateTime.of(2016, 1, 18, 18, 0,0, 0,
                        ZoneId.of(localTimeZone));
        assertEquals(computedTime, targetTime);
    }

    @Test
    public void convertNetTime2LocalZonedDatetimeDST() {
        long myTime = 635960808000000000L;
        String localTimeZone = "UTC";
        long localTimeMillis = ActigraphReader.GT3XfromTickToMillisecond(myTime);
        Instant computedTimeIns = Instant.ofEpochMilli(localTimeMillis);
        ZonedDateTime computedTime = computedTimeIns.atZone(ZoneId.of("UTC"));
        ZonedDateTime targetTime =
                ZonedDateTime.of(2016, 4, 12, 18, 0,0, 0,
                        ZoneId.of(localTimeZone));
        assertEquals(computedTime, targetTime);
    }

    @Test
    public void activityTimeMoveForward8Hours() {
        long myTime = 1271901600; // peudo-UNIX time
        String timeShift = "08:00:00";
        long targetTime = ActigraphReader.getTrueUnixTime(myTime, timeShift);
        assertEquals(myTime-8*60*60*1000, targetTime);
    }


    @Test
    public void activityTimeStayTheSame() {
        long myTime = 1271901600; // peudo-UNIX time
        String timeShift = "00:00:00";
        long targetTime = ActigraphReader.getTrueUnixTime(myTime, timeShift);
        assertEquals(myTime, targetTime);
    }


    @Test
    public void activityTimeMoveBack1Hour() {
        long myTime = 1271901600; // peudo-UNIX time
        String timeShift = "-01:00:00";
        long targetTime = ActigraphReader.getTrueUnixTime(myTime, timeShift);
        assertEquals(myTime+1*60*60*1000, targetTime);
    }
}

