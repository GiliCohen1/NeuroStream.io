import { ArrayMaxSize, ArrayMinSize, IsArray, IsNotEmpty, IsNumber, IsPositive, IsString, MaxLength } from 'class-validator';

export class TelemetryChunkDto {
  @IsString()
  @IsNotEmpty()
  @MaxLength(64)
  patientId!: string;

  /** Epoch milliseconds at which the chunk was captured. */
  @IsNumber({ allowNaN: false, allowInfinity: false })
  @IsPositive()
  timestamp!: number;

  @IsArray()
  @ArrayMinSize(16)
  @ArrayMaxSize(4096)
  @IsNumber({ allowNaN: false, allowInfinity: false }, { each: true })
  signalVector!: number[];
}
