import { Body, Controller, HttpCode, HttpStatus, Post } from '@nestjs/common';
import { TelemetryChunkDto } from './telemetry.dto';
import { TelemetryService } from './telemetry.service';

@Controller('api/v1/telemetry')
export class TelemetryController {
  constructor(private readonly telemetry: TelemetryService) {}

  @Post('stream')
  @HttpCode(HttpStatus.ACCEPTED)
  async stream(@Body() chunk: TelemetryChunkDto) {
    await this.telemetry.publish(chunk);
    return { accepted: true, patientId: chunk.patientId, samples: chunk.signalVector.length };
  }
}
